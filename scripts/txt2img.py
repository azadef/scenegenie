import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.guidance import Guider
import json, gc
#import pdb

#pdb.set_trace()

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="two raw eggs in a frying pan", #a painting of a virus monster playing guitar
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    
    parser.add_argument(
        "--begin",
        type=int,
        default=0
    )
    parser.add_argument(
        "--end",
        type=int,
        default=1024
    )
    
    parser.add_argument(
        "--data_source",
        type=str,
        default="coco" #prompt
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./scenegenie/predicted_1110/SG2IM_CLIP/COCO/"
    )
    parser.add_argument(
        "--scene_genie",
        action='store_true',
        help="use scene_genie guidance",
    )
    
    parser.add_argument(
        "--SDT2I_path",
        type=str,
        default="models/ldm/text2img-large/model.ckpt"
    )
        
    opt = parser.parse_args()


    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    print("Config loaded")
    model = load_model_from_config(config, opt.SDT2I_path)  # TODO: check path

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    prompt = opt.prompt
    score_corrector = None
    if opt.scene_genie:
        score_corrector = Guider(sampler)

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    
    
    if opt.data_source == "coco":
        folder = opt.data_path
        for index in range(opt.begin,  opt.end):
            obj_list = None
            with open(
                    f"{folder}/caption/"+ str(index) + "_caption", "r") as fp:

                prompt = json.load(fp)

            with open(
                    f"{folder}/obj_list/"+ str(index) + "_obj", "r") as f:

                obj_list = json.load(f)
                
            #diffusion_init = None
            bounding_box_pred = torch.load(
                f"{folder}/box/"
                + str(index) ).cuda()
            

            bounding_box_pred = torch.clamp(bounding_box_pred, min=0.0, max=1.0)

            obj_list.append(prompt)

            print("Starting creating sample " + str(index) + "...")
            print("obj_list", obj_list)

            print("prompt", prompt[0])


            bounding_box_pred =  torch.cat( (bounding_box_pred,torch.unsqueeze( torch.FloatTensor([0,0,1,1]), 0).cuda()),0).cpu()

            if opt.scene_genie:
                score_corrector.setData(prompt, bounding_box_pred, obj_list)
            
            gc.collect()
            
            with torch.no_grad():
                with model.ema_scope():
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(opt.n_samples * [""])
                    for n in trange(opt.n_iter, desc="Sampling"):
                        c = model.get_learned_conditioning(opt.n_samples * prompt)
                        shape = [4, opt.H//8, opt.W//8]
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         score_corrector=score_corrector)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                        for x_sample in x_samples_ddim:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{index:04}.png"))
                            base_count += 1

        
    else:
        all_samples=list()
        with torch.no_grad():
            with model.ema_scope():
                uc = None
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning(opt.n_samples * ["The eggs are raw. There are only two eggs."])
                for n in trange(opt.n_iter, desc="Sampling"):
                    c = model.get_learned_conditioning(opt.n_samples * [prompt])
                    shape = [4, opt.H//8, opt.W//8]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                     conditioning=c,
                                                     batch_size=opt.n_samples,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta,
                                                     score_corrector=score_corrector)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                    for x_sample in x_samples_ddim:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.png"))
                        base_count += 1
                    all_samples.append(x_samples_ddim)


        # additionally, save as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=opt.n_samples)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{prompt.replace(" ", "-")}.png'))

        print(f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy.")

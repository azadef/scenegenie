from typing import List, Tuple
from scipy import interpolate
import numpy as np
import torch
#import matplotlib.pyplot as plt
import abc
import clip
from torch.nn import functional as F
from torchvision import transforms

class GuideModel(torch.nn.Module, abc.ABC):
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def preprocess(self, x_img):
        pass

    @abc.abstractmethod
    def compute_loss(self, inp):
        pass


class MakeCutouts(torch.nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()

        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)

    
class Guider(torch.nn.Module):
    def __init__(self, sampler, guide_model=None, scale=1.0, verbose=False):
        """Apply classifier guidance

        Specify a guidance scale as either a scalar
        Or a schedule as a list of tuples t = 0->1 and scale, e.g.
        [(0, 10), (0.5, 20), (1, 50)]
        """
        super().__init__()
        self.sampler = sampler
        self.index = 0
        self.show = verbose
        clip_model, clip_preprocess = clip.load('ViT-B/32', device="cuda", jit=False) #was Vit-L/14  ViT-B/32
        clip_model.eval().requires_grad_(False)
        self.guide_model = clip_model
        self.history = []

        if isinstance(scale, (Tuple, List)):
            times = np.array([x[0] for x in scale])
            values = np.array([x[1] for x in scale])
            self.scale_schedule = {"times": times, "values": values}
        else:
            self.scale_schedule = float(scale)
        
        self.cutn = 16
        self.make_cutouts = MakeCutouts(clip_model.visual.input_resolution, self.cutn)
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        
        #self.ddim_timesteps = sampler.ddim_timesteps
        #self.ddpm_num_timesteps = sampler.ddpm_num_timesteps


    def get_scales(self):
        if isinstance(self.scale_schedule, float):
            return len(self.ddim_timesteps)*[self.scale_schedule]

        interpolater = interpolate.interp1d(self.scale_schedule["times"], self.scale_schedule["values"])
        fractional_steps = np.array(self.ddim_timesteps)/self.ddpm_num_timesteps
        return interpolater(fractional_steps)
    
    def setData(self, prompt, bounding_box, obj_list):
        self.prompt = prompt
        self.bounding_box = bounding_box
        self.obj_list = obj_list
        
        
    def cond_fn(self, x, t, obj_list = [], bounding_box= [], context=None, clip_embed=None, image_embed=None):
        with torch.enable_grad():
            x = x[:args.batch_size].detach().requires_grad_()
            n = x.shape[0]
            my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
            kw = {
                'context': context[:args.batch_size],
                'clip_embed': clip_embed[:args.batch_size] if model_params['clip_embed_dim'] else None,
                'image_embed': image_embed[:args.batch_size] if image_embed is not None else None
            }

            out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs=kw)
            fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
            x_in = out['pred_xstart'] * fac + x * (1 - fac)

            x_in /= 0.18215

            x_img = ldm.decode(x_in)
            # print("x_img sz", x_img.size())
            clip_in = normalize(make_cutouts(x_img.add(1).div(2)))
            # print("clip in", clip_in.size())
            clip_embeds = clip_model.encode_image(clip_in).float()
            # print("clip embeds sz", clip_embeds.size())
            dists = spherical_dist_loss(clip_embeds.unsqueeze(1), text_emb_clip.unsqueeze(0))
            # print("dists sz", dists.size())
            dists = dists.view([args.cutn, n, -1])
            # print("dists sz", dists.size())
            losses = dists.sum(2).mean(0)
            # print("losses size", losses.size())
            loss = losses.sum() * args.clip_guidance_scale

            weight_list = []
            for box in self.bounding_box[:-1]:
                weight = ( box[3] - box[1] ) * (box[2] - box[0]) 
                weight_list.append(weight)
            #print("weight list", weight_list)
            sum_weights = float(sum(weight_list))
            #print("sum", sum_weights)
            weight_list =np.array( weight_list) * (1/sum_weights)
            for (obj, box, weight) in zip(obj_list[:-1], bounding_box[:-1], weight_list):
                def create_new_patch(img, bounding_box, mode=None):
                    bounding_box = 256 * bounding_box
                    x = torch.randn(1, 3, 256, 256)
                    x[0:1, 0:3,
                    int(bounding_box[1]):int(bounding_box[3]),
                    int(bounding_box[0]):int(bounding_box[2])] = img[0:1, 0:3,
                    int(bounding_box[1]):int(bounding_box[3]),
                    int(bounding_box[0]):int(bounding_box[2])] 

                    return x

                x_box = create_new_patch(x_img, box)
                clip_img = self.normalize(make_cutouts(x_box.add(1).div(2))).to(device)

                clip_embeds_img = clip_model.encode_image(clip_img).float()

                text_obj = clip.tokenize(obj * args.batch_size, truncate=True).to(device)
                text_emb_clip_obj = clip_model.encode_text(text_obj)
                dists_obj = spherical_dist_loss(clip_embeds_img.unsqueeze(1), text_emb_clip_obj.unsqueeze(0))
                dists_obj = dists_obj.view([args.cutn, n, -1])
                losses_obj = dists_obj.sum(2).mean(0)

                #print("weight", weight)
                loss += losses_obj.sum() * args.clip_guidance_scale * weight
                #print("loss ", loss)

            grad = -torch.autograd.grad(loss, x)[0]

            return grad

                
    def spherical_dist_loss(self, x, y):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

    
    def clip_im_guidance(self, x_img, n, device="cuda"):
        clip_in = self.normalize(self.make_cutouts(x_img.add(1).div(2)))
        clip_embeds = self.guide_model.encode_image(clip_in).float()

        #prompt = "two raw eggs in a frying pan" * 4
        text = clip.tokenize(self.prompt, truncate=True).to(device)
        #text_clip_blank = clip.tokenize([args.negative]*args.batch_size, truncate=True).to(device)

        # clip context
        text_emb_clip = self.guide_model.encode_text(text)

        dists = self.spherical_dist_loss(clip_embeds.unsqueeze(1), text_emb_clip.unsqueeze(0))
        # print("dists sz", dists.size())
        dists = dists.view([self.cutn, n, -1])
        loss = dists.sum(2).mean(0)
        return loss
    
    
    def modify_score(self, model, e_t, x, t, c):
        self.ddim_timesteps = self.sampler.ddim_timesteps
        self.ddpm_num_timesteps = self.sampler.ddpm_num_timesteps
        #print("Begining guidance")
        # TODO look up index by t
        scale = self.get_scales()[self.index]
        device = "cuda"
        n = x.shape[0]
        
        if (scale == 0):
            return e_t

        sqrt_1ma = self.sampler.ddim_sqrt_one_minus_alphas[self.index].to(x.device)
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            pred_x0 = model.predict_start_from_noise(x_in, t=t, noise=e_t)
            x_img = model.first_stage_model.decode((1/0.18215)*pred_x0)
            
            bounding_box = self.bounding_box
            obj_list = self.obj_list
            #Azade
            #print(c)
            loss = self.clip_im_guidance(x_img, n)
            
            weight_list = []
            for box in bounding_box[:-1]:
                weight = ( box[3] - box[1] ) * (box[2] - box[0]) 
                weight_list.append(weight)
            #print("weight list", weight_list)
            sum_weights = float(sum(weight_list))
            #print("sum", sum_weights)
            weight_list =np.array( weight_list) * (1/sum_weights)
            for (obj, box, weight) in zip(obj_list[:-1], bounding_box[:-1], weight_list):
                def create_new_patch(img, bounding_box, mode=None):
                    bounding_box = 256 * bounding_box
                    x = torch.randn(1, 3, 256, 256)
                    x[0:1, 0:3,
                    int(bounding_box[1]):int(bounding_box[3]),
                    int(bounding_box[0]):int(bounding_box[2])] = img[0:1, 0:3,
                    int(bounding_box[1]):int(bounding_box[3]),
                    int(bounding_box[0]):int(bounding_box[2])] 

                    return x

                x_box = create_new_patch(x_img, box)
                clip_img = self.normalize(self.make_cutouts(x_box.add(1).div(2))).to(device)

                clip_embeds_img = self.guide_model.encode_image(clip_img).float()

                text_obj = clip.tokenize(obj * n, truncate=True).to(device)
                text_emb_clip_obj = self.guide_model.encode_text(text_obj)
                dists_obj = self.spherical_dist_loss(clip_embeds_img.unsqueeze(1), text_emb_clip_obj.unsqueeze(0))
                dists_obj = dists_obj.view([self.cutn, n, -1])
                losses_obj = dists_obj.sum(2).mean(0)

                #print("weight", weight)
                loss += losses_obj.sum() * weight
            
            #loss += clip_loss
            #inp = self.guide_model.preprocess(x_img)
            #loss = self.guide_model.compute_loss(inp)
            #Azade
            
            grads = torch.autograd.grad(loss.sum(), x_in)[0]
            correction = grads * scale

            if self.show:
                print(loss.item(), scale, correction.abs().max().item(), e_t.abs().max().item())
                self.history.append([loss.item(), scale, correction.min().item(), correction.max().item()])
                plt.imshow((inp[0].detach().permute(1,2,0).clamp(-1,1).cpu()+1)/2)
                plt.axis('off')
                plt.show()
                plt.imshow(correction[0][0].detach().cpu())
                plt.axis('off')
                plt.show()


        e_t_mod = e_t - sqrt_1ma*correction
        if self.show:
            fig, axs = plt.subplots(1, 3)
            axs[0].imshow(e_t[0][0].detach().cpu(), vmin=-2, vmax=+2)
            axs[1].imshow(e_t_mod[0][0].detach().cpu(), vmin=-2, vmax=+2)
            axs[2].imshow(correction[0][0].detach().cpu(), vmin=-2, vmax=+2)
            plt.show()
        self.index += 1
        if self.index == 50:
            self.index = 0
        return e_t_mod
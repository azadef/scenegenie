# SceneGenie: Scene Graph Guided Diffusion Models for Image Synthesis
[arXiv](https://arxiv.org/abs/2304.14573) | [BibTeX](#bibtex)


[**SceneGenie: Scene Graph Guided Diffusion Models for Image Synthesis**](https://arxiv.org/abs/2304.14573)<br/>



  
## Requirements
A suitable [conda](https://conda.io/) environment named `ldm` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate ldm
```

# Pretrained Models
A general list of all available checkpoints is available via [model zoo](#model-zoo).
If you use any of these models in your work, we are always happy to receive a [citation](#bibtex).

## Text-to-Image

Download the pre-trained weights (5.7GB)
```
mkdir -p models/ldm/text2img-large/
wget -O models/ldm/text2img-large/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```
and sample with
```
python scripts/txt2img.py --scene_genie --data_source coco --ddim_eta 0.0 --n_samples 4 --n_iter 4 --scale 5.0  --ddim_steps 50
```



## Comments 

- Our codebase builds heavily on [Stable Diffusion / LDM codebase](https://github.com/CompVis/stable-diffusion). 
Thanks for open-sourcing!

- The implementation of the transformer encoder is from [x-transformers](https://github.com/lucidrains/x-transformers) by [lucidrains](https://github.com/lucidrains?tab=repositories). 


## BibTeX

```
@inproceedings{farshad2023scenegenie,
  title={Scenegenie: Scene graph guided diffusion models for image synthesis},
  author={Farshad, Azade and Yeganeh, Yousef and Chi, Yu and Shen, Chengzhi and Ommer, B{\"o}jrn and Navab, Nassir},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={88--98},
  year={2023}
}


```



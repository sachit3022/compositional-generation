<div align="center">

# COIND: ENABLING LOGICAL COMPOSITIONS IN DIFFUSION MODELS

   <p style="font-size:1.2em">
      <a href="http://sachit3022.github.io"><strong>Sachit Gaudi</strong></a>·
      <a href="https://scholar.google.com/citations?user=mBrW_AkAAAAJ&hl=en&oi=ao"><strong>Gautam Sreekumar</strong></a>·
      <a href="https://scholar.google.com/citations?user=JKcrO9IAAAAJ&hl=en"><strong>Vishnu Boddeti</strong></a>
      <br>
      Michigan State University
      <br>
      <a href="https://openreview.net/forum?id=cCRlEvjrx4"><strong>ICLR 2025</strong></a>
</div>
   </p>



< Teaser Image and GIF of CelebA >  

## Installation
```bash
git clone https://github.com/sachit3022/compositional-generation.git
cd compositional-generation
conda env create -f environment.yml
conda activate compositional-generation
```

## CelebA
Download CelebA from ....
to speed up the training process run generation on the latent space.

Extract 

## Training
### Generative model
Example script to run the training based on the dataset setup and the CoInD regularizer
```bash
python coind/train.py --config-name=cmnist dataset=cmnist_partial diffusion.lambda_coind=1.0
```
This will create a folder called outputs/ and you can monitor the training via tensorboard or CSV generated in the output folder.
### Classifier for Confirmity score




### Inference
Once trained upload the checkpoint to checkpoints repository or download our checkpoints and place them in the checkpoints repository.
> wget 
#### Evaluation
Confirmity Score
To compute FID we use pytorch-fid to replicate the results please run the bash scripts of the evaluate/cmnist.sh
FID: we use pytorch-fid

#### Generate 
To have the guide to custom logical queries and fine grained control refer to our notbook.
### Custom dataset
To train on custom dataset follow our guide
#write a train_dataset and place it in the datasets/ folder

### Repository structure
Source code is present in CoInD repository. This repository consits of training module to train your models with CoInD built on lightning, huggingface and hydra for config management. 
  -  CoInD/
    - models ( Backbone for conditional denoising UNet / DiT ) 
    - inference ( Code to perform sampling with AND and OR operations)
    - score ( CoInD modules required for training: Loss, Noise schedule, EMA, ....)
    - datasets ( Code to load datasets in a respective format )
    - train.py ( training code integrating all the modules )
    - inference.py ( let's you control the generation and store the module)
    - evaluate
        - FID.py
        - confirmity_score.py
        - jsd.py
        - model_report.py
    - scripts
        - extract_latent.py ( To increase the speedup of the latent space diffsusion models, we pre-compute latents and store in the data folder )
  - config/ ( For config mangement we leverage hydra )


### Scripts for finetuning Stable Diffusion with CoInD

Coming soon ..... 

### CoInD 🤝 🤗

Coming soon ..... 



### Citation

If you find our work useful in your research, please consider starring the repo and citing:

```Bibtex
@inproceedings{gaudi2025coind,
   title={{CoInD: Enabling logical compositions in diffusion models}},
   author={Gaudi Sachit, Sreekumar Gautam, Boddeti Vishnu},
   booktitle={ICLR},
   year={2025}
}
```
For questions, feel free to post here or drop an email to this address- gaudisac@msu.edu
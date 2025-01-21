# COIND: ENABLING LOGICAL COMPOSITIONS IN DIFFUSION MODELS
< Teaser Image and GIF of CelebA >  

### Installation
> git clone https://github.com/sachit3022/compositional-generation.git
> conda env create -f environment.yml
> conda activate compositional-generation
#### CelebA
Download CelebA from ....
to speed up the training process run generation on the latent space.

Extract 

### Training
#### Generative model
#### Classifier for Confirmity score


### Inference
Once trained upload the checkpoint to checkpoints repository or download our checkpoints and place them in the checkpoints repository.
> wget 
#### Evaluation
evaluate.sh checkpoint < > dataset < > 

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
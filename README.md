<div align="center"> <div style="display: flex; justify-content: center; margin-bottom: 20px;"> <img src="https://iclr.cc/static/core/img/iclr-navbar-logo.svg" alt="ICLR 2025 Logo" style="width: 100px; height: auto;"> </div> <h1>COIND: ENABLING LOGICAL COMPOSITIONS IN DIFFUSION MODELS</h1> <p style="font-size:1.2em"> <a href="http://sachit3022.github.io"><strong>Sachit Gaudi</strong></a>· <a href="https://scholar.google.com/citations?user=mBrW_AkAAAAJ&hl=en&oi=ao"><strong>Gautam Sreekumar</strong></a>· <a href="https://scholar.google.com/citations?user=JKcrO9IAAAAJ&hl=en"><strong>Vishnu Boddeti</strong></a> <br> Michigan State University <br> <a href="https://openreview.net/forum?id=cCRlEvjrx4"><strong>ICLR 2025</strong></a> </p> </div>


<table>
  <tr>
    <td><img src="assets/full_2d_dataset.png" width="200"/></td>
    <td><img src="assets/train_dataset.png" width="200"/></td>
    <td><img src="assets/CoInD.png" width="200"/></td>
    <td><img src="assets/vanilla.png" width="200"/></td>
  </tr>
  <tr>
    <td><p align="center">Dataset present in nature where C<sub>1</sub>,C<sub>2</sub> are independent gaussian denoted on the respective axis.</td>
    <td><p align="center">Orthogonal Support (only few compositions seen in training)</p></td>
    <td><p align="center">Diffusion Model with CoInD generates unseen compositions</p></td>
    <td><p align="center">Vanilla Diffusion Models generate incorrect interpolations</p></td>
  </tr>
</table>

Complete walk through of the CoInD along with theoritical derivation for 2d gaussian is avaiable at [notebooks/2d_gaussian_generation.ipynb](notebooks/2d_gaussian_generation.ipynb)

### CoInD generates unseen compositions with precise control over attributes smile, and gender.
<table> <tr> <th style="width:100px; text-align:right; padding-right:10px;">CoInD</th> <td><img src="assets/1_coind.gif" width="200" align="top"></td> <td><img src="assets/2_coind.gif" width="200" align="top"></td> <td><img src="assets/3_coind.gif" width="200" align="top"></td> <td><img src="assets/4_coind.gif" width="200" align="top"></td> </tr> <tr> <th style="width:100px; text-align:right; padding-right:10px;">LACE</th> <td><img src="assets/1_lace.gif" width="200" align="top"></td> <td><img src="assets/2_lace.gif" width="200" align="top"></td> <td><img src="assets/3_lace.gif" width="200" align="top"></td> <td><img src="assets/4_lace.gif" width="200" align="top"></td> </tr> <tr> <th style="width:100px; text-align:right; padding-right:10px;">Composed GLIDE</th> <td><img src="assets/1_vanilla.gif" width="200" align="top"></td> <td><img src="assets/2_vanilla.gif" width="200" align="top"></td> <td><img src="assets/3_vanilla.gif" width="200" align="top"></td> <td><img src="assets/4_vanilla.gif" width="200" align="top"></td> </tr> </table>

For large vision dataset we give a sofisticated code where you can train large model on multiple GPUs. 
This repository is built on very cool, config management system called Hydra. Training from pytorch lightning, and model architectures from huggingface.
### Installation
```bash
git clone https://github.com/sachit3022/compositional-generation.git
cd compositional-generation
conda env create -f environment.yml
conda activate compositional-generation
```
### Download the datasets
- ColoredMNIST, Shapes3D will be automatically downloaded when you first call the training
- CelebA: Please refer to CelebA instructions to download the datasets.

Add the download location to the respective file in configs/datasets/*.yaml

## Training

### Classifier for Confirmity score
To measure the faithfullness of the generation we have introduced confirmity score, refer to Appendix of the paper.
```bash
python coind/cs_classifier/train.py --config-name=cs_cmnsit
```
### Diffusion in Latent Space
For CelebA dataset, we perform diffusion on Latent space. To speed up the training process run generation on the latent space. ( we borrow this from fast-DiT ) 
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=25670 coind/scripts/save_latent.py --encoder=vae --image-size=128 --dataset=celeba --data-path=/path/to/celeba --features-path=data/celeba
```

### Train Diffusion model
Modify the config of datasets( add /path/to/your/dataset) and callbacks CS(/path/to/your/checkpoint) or you can remove the callback. 
Example script to run the training based on the dataset setup and the CoInD regularizer.
```bash
python coind/train.py --config-name=cmnist dataset=cmnist_partial diffusion.lambda_coind=1.0
```
This will create a folder called outputs/ and you can monitor the training via tensorboard or CSV generated in the output folder.

### Inference
For Cmnist, CelebA datasets Once trained is completed, evaltion code is provided in the evaluate/ folder to find all the evaluation scripts. 
For CelebA first you have call the generate samples, which would generate samples, To evaluate CS and FID, refer to evaluate script in the evaluate/
If you donot want to train, we provide all the results along with te checkpoint in the follwing spreadsheet. You can download the checkpoint and evaluate.

#### Evaluation
Detail description of the metrics is provided in the paper.
- JSD
- Confirmity Score
- R2 Score
- Diversity
- FID
- Qualitative evaluations

<table>
  <tr>
    <th>Support</th>
    <th>Method</th>
    <th>Hyperparameter</th>
    <th>JSD</th>
    <th>CS AND</th>
    <th>CS NEG digit</th>
    <th>CS NEG color</th>
    <th>Joint</th>
    <th>H(digit)</th>
    <th>TRAIN AND</th>
    <th>TRAIN Joint</th>
  </tr>
  <tr>
    <td>uniform</td>
    <td>Lace</td>
    <td></td>
    <td></td>
    <td>0.7427</td>
    <td>0.9086</td>
    <td>0.6421</td>
    <td></td>
    <td>1.2161</td>
    <td>0.7264</td>
    <td></td>
  </tr>
  <tr>
    <td>uniform</td>
    <td>Vanilla</td>
    <td></td>
    <td>0.1594</td>
    <td>0.9790</td>
    <td>0.9967</td>
    <td>0.6922</td>
    <td>0.9997</td>
    <td>1.5220</td>
    <td>0.9777</td>
    <td>0.9999</td>
  </tr>
  <tr>
    <td>uniform</td>
    <td>CoInD</td>
    <td>0.2</td>
    <td>0.1353</td>
    <td>0.9961</td>
    <td>0.9544</td>
    <td>0.6732</td>
    <td>1.0000</td>
    <td>1.5866</td>
    <td>0.9973</td>
    <td>0.9999</td>
  </tr>
  <tr>
    <td>uniform</td>
    <td>CoInD</td>
    <td>1</td>
    <td>0.1042</td>
    <td>0.9996</td>
    <td>0.9816</td>
    <td>0.6639</td>
    <td>1.0000</td>
    <td>1.5972</td>
    <td>0.9999</td>
    <td>1.0000</td>
  </tr>
  <tr>
    <td>non-uniform</td>
    <td>Lace</td>
    <td></td>
    <td>0.9066</td>
    <td>0.6798</td>
    <td>0.9893</td>
    <td>0.8804</td>
    <td></td>
    <td>1.0508</td>
    <td>0.7877</td>
    <td></td>
  </tr>
  <tr>
    <td>non-uniform</td>
    <td>Vanilla</td>
    <td></td>
    <td>0.2984</td>
    <td>0.8054</td>
    <td>0.8749</td>
    <td>0.6072</td>
    <td>0.9999</td>
    <td>0.8387</td>
    <td>0.9025</td>
    <td>0.9999</td>
  </tr>
  <tr>
    <td>non-uniform</td>
    <td>CoInD</td>
    <td>1</td>
    <td>0.1479</td>
    <td>0.9993</td>
    <td>0.9596</td>
    <td>0.6164</td>
    <td>1.0000</td>
    <td>1.3363</td>
    <td>0.9993</td>
    <td>0.9999</td>
  </tr>
  <tr>
    <td>partial</td>
    <td>Lace</td>
    <td></td>
    <td></td>
    <td>0.1536</td>
    <td>0.0629</td>
    <td>0.2885</td>
    <td></td>
    <td>0.6497</td>
    <td>0.9883</td>
    <td></td>
  </tr>
  <tr>
    <td>partial</td>
    <td>Vanilla</td>
    <td></td>
    <td>2.7500</td>
    <td>0.1270</td>
    <td>0.0486</td>
    <td>0.3341</td>
    <td>0.4793</td>
    <td>0.6360</td>
    <td>0.9853</td>
    <td>0.9988</td>
  </tr>
  <tr>
    <td>partial</td>
    <td>CoInD</td>
    <td>0.5</td>
    <td>1.1976</td>
    <td>0.4586</td>
    <td>0.5204</td>
    <td>0.5646</td>
    <td>0.6771</td>
    <td>0.6876</td>
    <td>0.9992</td>
    <td>1.0000</td>
  </tr>
  <tr>
    <td>partial</td>
    <td>CoInD</td>
    <td>1</td>
    <td>1.1736</td>
    <td>0.5444</td>
    <td>0.5386</td>
    <td>0.5192</td>
    <td>0.3742</td>
    <td>0.7654</td>
    <td>0.9994</td>
    <td>0.9994</td>
  </tr>
  <tr>
    <td>partial</td>
    <td>CoInD</td>
    <td>5</td>
    <td>0.6615</td>
    <td>0.0295</td>
    <td>0.0356</td>
    <td>0.0425</td>
    <td>0.0235</td>
    <td>1.2299</td>
    <td>0.9987</td>
    <td>0.9986</td>
  </tr>
  <tr>
    <td>partial</td>
    <td>CoInD</td>
    <td>10</td>
    <td>0.5521</td>
    <td>0.0170</td>
    <td>0.0234</td>
    <td>0.1035</td>
    <td>0.0176</td>
    <td>1.3785</td>
    <td>0.9978</td>
    <td>0.9979</td>
  </tr>
</table>

Similarly, for the other datasets please refer to <a href="https://docs.google.com/spreadsheets/d/1lHcqRJTo6JgRHh_PwHbt3MIp-PmSKPnqU4wTS9YXB8M/edit?usp=sharing">https://docs.google.com/spreadsheets/d/1lHcqRJTo6JgRHh_PwHbt3MIp-PmSKPnqU4wTS9YXB8M/edit?usp=sharing</a> </p> </div>

For celeba, you can explore the precise control by downloading the checkpoints from the above google sheets, and run  [notebooks/celeba_control_smile.ipynb](notebooks/celeba_control_smile.ipynb)




### Custom Datatset, logic, constraints

To train on custom dataset follow our guide
#write a train_dataset and place it in the datasets/ folder

Checkout score/sampling.py file it is built on 



### Scripts for finetuning Stable Diffusion with CoInD

| CoInD | ![Image](assets/sd_results/male_smile_1_coind.jpg) | ![Image](assets/sd_results/male_smile_2_coind.jpg) | ![Image](assets/sd_results/male_smile_3_coind.jpg) |
|-------|----------------------------|----------------------------|----------------------------|
| Composed GLIDE | ![Image](assets/sd_results/male_smile_1_vanilla.jpg) | ![Image](assets/sd_results/male_smile_2_vanilla.jpg) | ![Image](assets/sd_results/male_smile_3_vanilla.jpg) |

Code for finetuning scripts is available in sd_finetune_coind/ Most of it is borrowed from the huggingface finetuning scripts.

### CoInD 🤝 🤗

Coming soon ..... 

### We borrow code from multiple resources, attribution is given in the code.
Special mention to Mushrur, huggingface, hydra, pytorch lightning, lucidrains.


### Utilty of CoInD
ICLRW Synthetic data workshop: Compositional World Knowledge leads to High Utility Synthetic data
To run the code follow the process above, the only change will be in the evaluation on Compositional Generalization task
```bash
python /coind/evaluate/evaluate_synthetic_data.py --sythetic_data_path=/path/to/synthetic_data --sythetic_data_path=/path/to/originaldata --train_on=synthetic 
```

### Citation

If you find our work useful in your research, please consider starring the repo and citing:

```Bibtex
@inproceedings{
  gaudi2025coind,
  title={CoInD: Enabling Logical Compositions in Diffusion Models},
  author={Sachit Gaudi and Gautam Sreekumar and Vishnu Boddeti},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=cCRlEvjrx4}
}
```
For questions, feel free to post here or drop an email to this address- gaudisac@msu.edu

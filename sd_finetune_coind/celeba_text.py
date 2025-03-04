import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import os
import json
from PIL import Image
from torchvision import transforms
import random
from torchvision.transforms.functional import crop
import numpy as np
from PIL import ImageFile, ImageDraw
import pandas as pd
import torch

class LoadImages(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        y = torch.tensor([1,1])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'text': y}

# Define transformation function using the dictionary

class CelebaDataset():
    def __init__(self,dataset_dir, split='train',clip_preprocess=None,accelerator=None,transform='text'):

        self.dataset_dir = dataset_dir

        self.txt = f"{self.dataset_dir}/list_attr_celeba.txt" #os.path.join(self.dataset_dir, 'metadata_orgimg_v4.json')
        self.split_file = f"{self.dataset_dir}/list_eval_partition.txt"
        #if transforms as torch.transforms.Compose do something
        if isinstance(transform,transforms.Compose):
            self.transform = transform
            transform = 'identity'
        else:
            self.transform = None       
        samples = self.transform_gender_smile([self.txt,self.split_file],split,transform)
        random.shuffle(samples)
        samples_new =[]

        for i,data in enumerate(samples):
            if os.path.exists(f"{self.dataset_dir}/Img/img_align_celeba/"+data[0].split(',')[0]):
                samples_new.append(data)
            # if i > self.max_length:
            #     break
        self.samples = samples_new


    def __getitem__(self, idx):
        sample = self.samples[idx]
        target_images = Image.open(f"{self.dataset_dir}/Img/img_align_celeba/"+sample[0].split(',')[0])

        if not target_images.mode == "RGB":
            target_images = target_images.convert("RGB")
        
        y = sample[1]
        if self.transform:
            target_images = self.transform(target_images)
            y = torch.tensor(sample[1])
        
        example = {'image': target_images, 'text':y}
        return example

    def __len__(self):
        return len(self.samples)

    def __call__(self):
        for x in self:
            yield x
    
    @property
    def photo_descriptions(self):
        return {
            (1, 1): 'Photo of a smiling male celebrity',
            (1, 0): 'Photo of a non-smiling male celebrity',
            (0, 1): 'Photo of a smiling female celebrity',
            (0, 0): 'Photo of a non-smiling female celebrity',
            (1, 2): 'Photo of a male celebrity',
            (0, 2): 'Photo of a female celebrity',
            (2, 1): 'Photo of a smiling celebrity',
            (2, 0): 'Photo of a non-smiling celebrity',
            (2, 2): 'Photo of a celebrity'
        }
        
    def text_transform(self,row):
        # 20% chance of making row['Male'] = 2 and row['Smiling'] = 2
        if random.random() < 0.2:
            row['Male'] = 2
        if random.random() < 0.2:
            row['Smiling'] = 2
        return self.photo_descriptions[(row['Male'], row['Smiling'])]

    def identity_transform(self,row):
        return (row['Male'], row['Smiling'])

    def transform_gender_smile(self,input_files,split,transform='text'):
        # Define the column names based on the provided format
        columns = ["image_id", "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs",
                "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby",
                "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male",
                "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
                "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
                "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]

        # Read the data from the input file
        data_file,splits_file = input_files
        df = pd.read_csv(data_file, delimiter=',')#pd.read_csv(input_file, delim_whitespace=True, header=None, names=columns)
        split_df = pd.read_csv(splits_file, delimiter=' ', header=None, names=['image_id', 'split'])
        train_val_test = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        # based on data filter df to only include the split
        df = df.merge(split_df, on='image_id')
        df = df[df['split'] == train_val_test[split]]
        
        # Extract only the 'Male' and 'Smiling' columns and convert them to binary vectors
        df['Male'] = df['Male'].apply(lambda x: 1 if x == 1 else 0)
        df['Smiling'] = df['Smiling'].apply(lambda x: 1 if x == 1 else 0)

        #there should be no male and smiling
        df = self.filter_data(df)

        # Apply the transformation function to the data
        if transform == 'text':
            df['Male_Smiling_Simplified'] = df.apply(self.text_transform, axis=1)
        elif transform == 'identity':
            df['Male_Smiling_Simplified'] = df.apply(self.identity_transform, axis=1)
        else:
            raise ValueError("Invalid transformation function")
            
        # Convert the list of lists into a list of tuples (image_id, tensor)
        tensors = [(row['image_id'], row['Male_Smiling_Simplified']) for _, row in df.iterrows()]
        return tensors
    def filter_data(self,df):
        return df[~((df['Male'] == 1) & (df['Smiling'] == 1))]

def image_grid(imgs, rows, cols,captions=None):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    if captions:
        if len(captions) != len(imgs):
            raise ValueError("Number of captions should be equal to number of images")
        for i, img in enumerate(imgs):
            grid.paste(img, box=(i%cols*w, i//cols*h))
            draw = ImageDraw.Draw(grid)
            draw.text((i%cols*w, i//cols*h), captions[i], fill='purple')
    else:
        for i, img in enumerate(imgs):
            grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


class CelebafullDataset(CelebaDataset):
    def filter_data(self,df):
        return df[((df['Male'] == 1) & (df['Smiling'] == 1))]

if __name__ == "__main__":
    # #plot the grid of images with the text as title
    train_data_dir="/research/hal-datastore/datasets/original/celeba/"
    # dataset = CelebaDataset(train_data_dir, split='train',clip_preprocess=None,accelerator=None)
    # images= [dataset[i]['image'] for i in range(16)]
    # captions = [dataset[i]['text'] for i in range(16)]

    # grid = image_grid(images, 4, 4,captions)
    # grid.save('images/sample_celeba.png')
    data_dir = "/research/hal-datastore/datasets/original/celeba/"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = CelebafullDataset(train_data_dir, split='train', transform=transform)
    count = 0
    for i in range(1000):
        if dataset[i]['text'][0] == 1 and dataset[i]['text'][1] == 1:
            count += 1
    print(count)

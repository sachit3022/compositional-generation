import os
import json
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np



class SytheticData(Dataset):
    def __init__(self,root_dir,transform=None):
        self.root_dir = root_dir
        metadata = os.path.join(root_dir,"metadata.json")
        with open(metadata) as f:
            self.metadata = [json.loads(x) for x in f.readlines()]
        self.transform = transform
    def __len__(self):
        return len(self.metadata)
    def __getitem__(self, idx):
        #load the image and query and null token
        filename = self.metadata[idx]['filename']
        # last_int = int(filename.split("_")[-1].replace(".png",""))
        # filename = filename.replace(f"_{last_int}.png",f"_{last_int:06d}.png")
        img_name = os.path.join(self.root_dir,"gen",filename)
        #if .png only
        if ".png" in img_name:
            image = Image.open(img_name)
        else:
            image = torch.tensor(np.load(img_name))
        if self.transform:
            image = self.transform(image)
        query = torch.tensor(self.metadata[idx]['query'])
        null_token = torch.tensor(self.metadata[idx]['null_token'])
        return {"X":image, "label":query,"idx":idx}
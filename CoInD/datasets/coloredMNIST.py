# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
from PIL import Image
import os
import random
from dotmap import DotMap

# transforms.Normalize((0.4914,0.4822,0.4465),
#                                                    (0.2023,0.1994,0.2010)),


normalized_rgbs = [
    (0.18, 0.31, 0.31),  # Dark Slate Gray
    (0.13, 0.55, 0.13),  # Forest Green
    (1.0, 0.27, 0.0),    # Orange Red
    (1.0, 1.0, 0.0),     # Yellow
    (0.78, 0.08, 0.52),  # Medium Violet Red
    (0.0, 1.0, 0.0),     # Lime
    (0.25, 0.41, 0.88),  # Royal Blue
    (0.0, 1.0, 1.0),     # Aqua
    (0.0, 0.0, 1.0),     # Blue
    (1.0, 0.87, 0.68)    # Navajo White
]

COLOURS =  torch.tensor(normalized_rgbs)

class Diagonal:
    def __init__(self,train=True,random_color=False):
        self.train = train
        self.random_color = random_color
    def __call__(self, y):
        if self.random_color:
            return random.randint(0,9)
        elif self.train:
            return random.randint(y,min(y+1,9))
        else:
            return random.choice([i for i in range(10) if i not in [y,min(y+1,9)]])

class SpuriousDiagonal:
    def __init__(self,train=True,random_color=False):
        self.train = train
        self.random_color = random_color
    def __call__(self, y):
        if self.random_color:
            return random.randint(0,9)
        elif self.train:
            if random.uniform(0,1) < 0.5:
                return random.randint(y,min(y+1,9))
            else:
                return random.randint(0,9)
        else:
            return random.randint(0,9) 

class ColoredMNISTGaussian(Dataset):
    def __init__(self, root_dir,train=True, download=False,random_color= False):
        self.mnist = datasets.MNIST(root=root_dir, train=train, transform=transforms.ToTensor(), download=download)    
        self.colors = COLOURS
        #create a gaussian distribution for the color 
        if train:
            categorical_probs = [0.0010, 0.0076, 0.0360, 0.1095, 0.2132, 0.2663, 0.2132, 0.1095, 0.0360, 0.0076]
        else:
            categorical_probs = [0.1]*10


        categorical_probs = np.array(categorical_probs)
        categorical_probs /= categorical_probs.sum()
        self.randlabels = np.random.choice(len(categorical_probs), size=len(self.mnist), p=categorical_probs)
        #also sample the digit label from the same distribution

        count_samples = categorical_probs*len(self.mnist)
        count_samples = count_samples.astype(int)
        new_target_idx = []
        for i in range(10):
            idx = np.where(self.mnist.targets == i)[0]
            idx = np.random.choice(idx,count_samples[i],replace=True)
            new_target_idx.extend(idx.tolist())
            
        self.mnist.targets = self.mnist.targets[new_target_idx]
        self.mnist.data = self.mnist.data[new_target_idx]


        self.transform = None
        

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        colour_idx = self.randlabels[idx]
        colour = self.colors[colour_idx]
        coloured_img = self.colour_image(img, colour)

        if self.transform:
            coloured_img = self.transform(coloured_img)

        return {'X': coloured_img, 'label':  torch.tensor([label,colour_idx]).long()}
    
    def colour_image(self, img, colour):
        #color all the pixcels that are not black
        img = img.squeeze(0)
        color_indices = img > 0
        img = img.repeat(3, 1, 1) # convert to 3 channel image
        img[:, color_indices] = colour[:, None]
        return img

class ColoredMNIST1d(Dataset):
    def __init__(self, root_dir,train=True, download=False,random_color= False):
        self.mnist = datasets.MNIST(root=root_dir, train=train, transform=transforms.ToTensor(), download=download)    
        self.bias = Diagonal(train,random_color)
        self.colors = COLOURS
        self.randlabels = np.vectorize(self.bias)(self.mnist.targets.numpy())
        self.transform = None
        
    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        colour_idx = self.randlabels[idx]
        colour = self.colors[colour_idx]
        coloured_img = self.colour_image(img, colour)

        if self.transform:
            coloured_img = self.transform(coloured_img)

        return {'X': coloured_img, 'label':  torch.tensor([label,colour_idx]).long()}
    
    def colour_image(self, img, colour):
        #color all the pixcels that are not black
        img = img.squeeze(0)
        color_indices = img > 0
        img = img.repeat(3, 1, 1) # convert to 3 channel image
        img[:, color_indices] = colour[:, None]
        return img

class ColoredMNISTSpurious(Dataset):
    def __init__(self, root_dir,train=True, download=False,random_color= False):
        self.mnist = datasets.MNIST(root=root_dir, train=train, transform=transforms.ToTensor(), download=download)    
        self.bias = SpuriousDiagonal(train,random_color)
        self.colors = COLOURS
        self.randlabels = np.vectorize(self.bias)(self.mnist.targets.numpy())
        self.transform = None
        

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        colour_idx = self.randlabels[idx]
        colour = self.colors[colour_idx]
        coloured_img = self.colour_image(img, colour)

        if self.transform:
            coloured_img = self.transform(coloured_img)

        return {'X': coloured_img, 'label':  torch.tensor([label,colour_idx]).long()}
    
    def colour_image(self, img, colour):
        #color all the pixcels that are not black
        img = img.squeeze(0)
        color_indices = img > 0
        img = img.repeat(3, 1, 1) # convert to 3 channel image
        img[:, color_indices] = colour[:, None]
        return img


class ColoredMNIST4d(Dataset):
    def __init__(self, root_dir,train=True, download=False,random_color= False):
        self.mnist = datasets.MNIST(root=root_dir, train=train, transform=transforms.ToTensor(), download=download)    
        self.bias = Diagonal(train,random_color)
        self.colors = COLOURS
        self.randlabels = np.vectorize(self.bias)(self.mnist.targets.numpy())
        self.transform = None
        self.train = train
        self.random_color = random_color
        
    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        if self.train and not self.random_color:
            std = 0.2
            colour_idx = self.randlabels[idx]
            colour = self.colors[colour_idx] + torch.randn(3)*std
            colour = torch.clamp(colour,0,1)

        else:
            colour = torch.tensor([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)])

        coloured_img = self.colour_image(img, colour)

        if self.transform:
            coloured_img = self.transform(coloured_img)
        
        #round the color to nerest 0.1
        colour = torch.round(colour*10).long()
        return {'X': coloured_img, 'label':  torch.tensor([label]+colour.tolist()).long()}
    
    def colour_image(self, img, colour):
        #color all the pixcels that are not black
        img = img.squeeze(0)
        color_indices = img > 0
        img = img.repeat(3, 1, 1) # convert to 3 channel image
        img[:, color_indices] = colour[:, None]
        return img


class ColoredMNIST(Dataset):
    def __init__(self,option):
        self.data_split = option.data_split
        data_dic = np.load(os.path.join(option.data_dir,'mnist_10color_jitter_var_%.03f.npy'%option.color_var),encoding='latin1',allow_pickle=True).item()
        if self.data_split == 'train':
            self.image = data_dic['train_image']
            self.label = data_dic['train_label']
        elif self.data_split == 'test':
            self.image = data_dic['test_image']
            self.label = data_dic['test_label']
        elif self.data_split == 'full':
            #join the train and test data
            self.image = np.concatenate([data_dic['train_image'],data_dic['test_image']],axis=0)
            self.label = np.concatenate([data_dic['train_label'],data_dic['test_label']],axis=0)


        color_var = option.color_var
        self.color_std = color_var**0.5

        self.T = transforms.Compose([
                              transforms.ToTensor()
                              # transforms.Normalize((0.4914,0.4822,0.4465),
#                                                    (0.2023,0.1994,0.2010)),
                                    ])

        self.ToPIL = transforms.Compose([
                              transforms.ToPILImage(),
                              ])



    def __getitem__(self,index):
        label = self.label[index]
        image = self.image[index]

        
        image = self.ToPIL(image)


        label_image = image.resize((14,14), Image.NEAREST) 

        label_image = torch.from_numpy(np.transpose(np.array(label_image),(2,0,1)))
        mask_image = torch.lt(label_image.float()-0.00001, 0.) * 255
        label_image = torch.div(label_image,32)
        label_image = label_image + mask_image
        
        label_image = label_image.long()


        #print(self.T(image).dtype,torch.unique(label_image.view(3,-1),dim=1,sorted=True)[:,0].dtype,label.astype(np.int64).dtype)
        #changed only the label_image part
        s = torch.unique(label_image.view(3,-1),dim=1,sorted=True)[:,0]
        label = torch.cat([torch.tensor([label]),s]).long()

        return {"X":self.T(image),"label":label}

        

    def __len__(self):
        return self.image.shape[0]



class CounterfactualColoredMNIST(Dataset):
    def __init__(self,counterfactual_option,train=True):
        self.X, self.y, self.s = [],[],[]
        for loc in counterfactual_option:
            loc_x = os.path.join(loc,'generated_samples.pt')
            loc_y = os.path.join(loc,'generated_labels.pt')
            loc_s = os.path.join(loc,'generated_sensitive.pt')
            self.X.append(torch.load(loc_x))
            self.y.append(torch.load(loc_y))
            self.s.append(torch.load(loc_s))
        self.X = torch.cat(self.X)
        self.y = torch.cat(self.y)
        self.s = torch.cat(self.s)

    def __getitem__(self,index):
        #print dtyoes of X,y,s
        #print(self.X[index].dtype,self.y[index].dtype,self.s[index].dtype)
        return {"X":self.X[index], "sensitive":self.s[index],  "label":self.y[index].item()}
        
    def __len__(self):
        return self.X.shape[0]

    def inverse_transform(self, X):
        return X
    
class MixInDataset(ColoredMNIST):
    def __init__(self, option,counterfactual_option, train=True,mix_in=0.5):
        super().__init__(option, train)
        self.counterfactual = CounterfactualColoredMNIST(counterfactual_option, train,)
        self.mix_in =mix_in

    def __getitem__(self, idx):
        if random.uniform(0, 1) <= self.mix_in:
            return super().__getitem__(idx)
        else:
            return self.counterfactual[idx]


    def __len__(self):
        return min(self.image.shape[0], len(self.counterfactual))



if __name__ == "__main__":
    args = DotMap({
        'data_dir': '/research/hal-gaudisac/Diffusion/controllable-generation/data/colored_mnist',
        'data_split': 'train',
        'color_var': 0.02
    })

    dataset = ColoredMNIST(args)
    print(len(dataset))
    print(dataset[0]['X'].shape)
    print(dataset[0]['label'])
    print(dataset[0]['label'].shape)
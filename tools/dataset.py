import  torch
import  os, glob
import  random, csv

from    torch.utils.data import Dataset, DataLoader

from    torchvision import transforms
from    PIL import Image
import numpy as np


class Dataset_train(Dataset):
    def __init__(self, data_all,mode='false'):

        self.noise_list = data_all[0]



    def __len__(self):
        return len(self.noise_list)

    def __getitem__(self, idx):
        data = self.noise_list[idx]

        noise = torch.Tensor(data)
        return  noise


class Dataset_val(Dataset):
    def __init__(self, data_all):

        noises, labels = data_all[0], data_all[1]
        assert len(noises) == len(noises)
        self.noise_list = noises
        self.labels = labels

    def __len__(self):
        return len(self.noise_list)

    def __getitem__(self, idx):
        data = self.noise_list[idx]



        m1 = self.labels[idx].split("_")[0]
        m2 = self.labels[idx].split("_")[1]
        n1 = self.labels[idx].split("_")[2]
        n2 = self.labels[idx].split("_")[3]
        noise =torch.Tensor(data)
        return noise, int(m1),int(m2),int(n1),int(n2)





def main():
    root = r"C:\Users\CT\Desktop\autoencoder\patch\NC2016_2564"

    # train_dataset = Dataset_val(root)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    # for batchidx,(img, noise,label,m1,m2,n1,n2) in enumerate(train_dataloader):
    #     print(m1.shape)
    train_dataset = Dataset_train(root)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    for batchidx, (img, noise, label) in enumerate(train_dataloader):
        print(img.shape)








if __name__ == '__main__':
    main()
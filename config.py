import torch
from torchvision import transforms
import numpy as np


class Config:
    def __init__(self):

        self.MisakaNum = "10032"
        self.batchsz = 128
        self.dataset_name = "tiny-imagenet-200"
        self.classNum = 200

        self.trainDir = "./../Dataset/"+self.dataset_name+"/subtrain"
        self.validDir = "./../Dataset/" + self.dataset_name + "/valid"
        self.testDir = "./../Dataset/"+self.dataset_name+"/test"

        self.barlowtwins_path = "./../barlowtwins.pth"
        self.model_path = "./model/" + self.MisakaNum + ".pth"

        self.targetCenters = torch.as_tensor(
            np.genfromtxt("./fixparam/" + self.dataset_name + "_pca" + str(self.classNum - 1) + "_t_centers.csv",
                          dtype=float, delimiter=",", skip_header=0))
        self.bias = torch.as_tensor(
            np.genfromtxt("./fixparam/" + self.dataset_name + "_pca" + str(self.classNum - 1) + "_t_mu.csv",
                          dtype=float, delimiter=",", skip_header=0))
        self.coeff = torch.as_tensor(
            np.genfromtxt("./fixparam/" + self.dataset_name + "_pca" + str(self.classNum - 1) + "_t_coeff.csv",
                          dtype=float, delimiter=",", skip_header=0))

        self.train_transformer = transforms.Compose([
            # transforms.Resize([32, 32]),
            transforms.Resize([224, 224]),
            transforms.RandomCrop([224, 224], padding=28, pad_if_needed=False, fill=0, padding_mode='constant'),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # imagenet
                                 std=[0.229, 0.224, 0.225])
        ])

        self.test_transformer = transforms.Compose([
            # transforms.Resize([32, 32]),
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # imagenet
                                 std=[0.229, 0.224, 0.225])
        ])

        self.USE_MULTI_GPU = True

        self.optimizer = "AdamW"
        self.lr = 3e-4

        self.epoch_num = 50




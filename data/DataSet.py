from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def getDataFromFile(dir, thisTransformer, config):
    dataset = ImageFolder(dir, transform=thisTransformer)  # data_dir精确到分类目录的上一级
    thisDataloader = DataLoader(dataset, batch_size=config.batchsz, shuffle=True, num_workers=4, prefetch_factor=1)
    return thisDataloader

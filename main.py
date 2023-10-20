from config import Config
from sys import path

path.append('./data')
path.append('./model')
path.append('./tool')

from data.DataSet import getDataFromFile
from model.modelseting import MisakaNet
from model.training import finetuning
from log.LogCtrl import Misakalog
import os



def main():
    cfg = Config()
    log = Misakalog(cfg)

    train_dataloader = getDataFromFile(cfg.trainDir, cfg.train_transformer, cfg)
    valid_dataloader = getDataFromFile(cfg.validDir, cfg.test_transformer, cfg)
    test_dataloader = getDataFromFile(cfg.testDir, cfg.test_transformer, cfg)

    misaka = MisakaNet(cfg)
    finetuning(cfg, misaka, train_dataloader, valid_dataloader, test_dataloader, log)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'  # 如果是多卡改成类似0,1,2
    main()




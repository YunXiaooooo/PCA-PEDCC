from sys import path

import numpy
import torch
import numpy as np
# path.append('./../data')
# path.append('./../model')
# path.append('./../tool')

from config import *
from data.DataSet import *
from model.modelseting import *
import pandas as pd




def interface_main():

    for i in range(2, 3):
        cfg = Config()

        misaka = MisakaNet(cfg)
        misaka = torch.nn.DataParallel(misaka).cuda()
        load_trained_misaka(misaka, cfg)
        print(misaka)
        Dataset = ImageFolder(cfg.trainDir, transform=cfg.train_transformer)  # data_dir精确到分类目录的上一级
        dataloader = DataLoader(Dataset, batch_size=cfg.batchsz, shuffle=False)
        interface_and_get_feature(misaka, cfg, dataloader, dataType="train", fi=i)



def interface_and_get_feature(model, config, input_data, dataType, fi):
    feature_data = np.array([])
    label_data = np.array([])
    norm_data = np.array([])

    model.cuda()
    model.eval()
    with torch.no_grad():
        # interface
        for x, label in input_data:
            x, label = x.cuda(), label.cuda()

            xout, xnorm = model(x)


            label_array = label.cpu().numpy()
            # label_array = numpy.ones(label.shape)*config.main_c
            if label_data.size == 0:
                label_data = label_array
            else:
                label_data = np.concatenate((label_data, label_array), axis=0)

            norm_array = xnorm.squeeze().cpu().numpy()
            if norm_data.size == 0:
                norm_data = norm_array
            else:
                norm_data = np.concatenate((norm_data, norm_array), axis=0)

            features_array = xout.cpu().numpy()
            if feature_data.size == 0:
                feature_data = features_array
            else:
                feature_data = np.concatenate((feature_data, features_array), axis=0)

            print("label_data")
            print(label_data)

        print("feature_data.shape = ", feature_data.shape)
        print("label_data.shape = ", label_data.shape)
        fl_data = np.insert(feature_data, feature_data.shape[1], label_data, axis=1)
        # fl_data = np.insert(fl_data, fl_data.shape[1], norm_data, axis=1)

        print("fl_data.shape", fl_data.shape)
        np.savetxt('./../Dataset/features/cifar100/' + dataType + '_by_' + 'Misaka' + config.MisakaNum +'(' + str(fi) + ')' +'.csv',
                   fl_data, delimiter=',', fmt='%f')


def load_trained_misaka(model, config):
    if os.path.exists(config.model_path):
        checkpoint = torch.load(config.model_path)
        model.load_state_dict(checkpoint['model'])
        print('load trained {} success! '.format(config.MisakaNum))
    else:
        print('No  trained model  \'{}\''.format(config.MisakaNum))
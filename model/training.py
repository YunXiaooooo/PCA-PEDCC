import  torch
from torch import nn, optim
from model.modelseting import model_state_save, model_state_load
from tqdm import tqdm


def finetuning(config, model, train_data, valid_data, test_data, misakalog):
    if config.USE_MULTI_GPU:
        model = torch.nn.DataParallel(model).cuda()  # 多卡
    else:
        model = model.cuda()  # 单卡
    misakalog.printInfo(model)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.001)

    start_epoch, valid_best_acc = model_state_load(config.model_path, model, optimizer)
    end_epoch = config.epoch_num

    for epoch in tqdm(range(start_epoch, end_epoch), ncols=50, leave=True):
        model.eval()

        train_loss = torch.tensor([0]).cuda()
        train_sampleNum = 0
        train_correct = 0
        for batchidx, (x, label) in enumerate(train_data):
            # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
            x, label = x.cuda(), label.cuda()
            xout = model(x)

            loss = cosineLossFunc(xout, label, config.classNum)

            # loss = loss1+0.01*epoch*loss2
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                # loss 累加
                train_loss = train_loss + loss.detach()
                pred = xout.argmax(dim=1)
                correct = torch.eq(pred, label).float().sum().item()
                train_correct += correct
                train_sampleNum += x.shape[0]

        with torch.no_grad():
            train_acc = train_correct/train_sampleNum
            str = '%d:loss=%.5f, trainAcc=%.5f' %(epoch, train_loss.item(), train_acc)

        model.eval()  # 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化（下面是推理预测，不是训练）
        with torch.no_grad():
            # valid
            valid_loss = torch.tensor([0]).cuda()
            valid_correct = 0
            SampleNum = 0
            for x, label in valid_data:
                x, label = x.cuda(), label.cuda()
                xout = model(x)
                loss = cosineLossFunc(xout, label, config.classNum)
                # loss 累加
                valid_loss = valid_loss + loss.detach()
                pred = xout.argmax(dim=1)
                correct = torch.eq(pred, label).float().sum().item()
                valid_correct += correct
                SampleNum += x.shape[0]

            valid_acc = valid_correct / SampleNum
            str += ', validLoss=%.5f, validAcc=%.5f' % (valid_loss.item(), valid_acc)

            # test
            test_loss = torch.tensor([0]).cuda()
            test_correct = 0
            SampleNum = 0
            for x, label in test_data:
                x, label = x.cuda(), label.cuda()
                xout = model(x)
                loss = cosineLossFunc(xout, label, config.classNum)
                # loss 累加
                test_loss = test_loss + loss.detach()
                pred = xout.argmax(dim=1)
                correct = torch.eq(pred, label).float().sum().item()
                test_correct += correct
                SampleNum += x.shape[0]

            test_acc = test_correct / SampleNum
            str += ', testLoss=%.5f, testAcc=%.5f' % (test_loss.item(), test_acc)

            if valid_acc >= valid_best_acc:
                model_state_save(model, config.model_path, optimizer, epoch, valid_acc)
                valid_best_acc = valid_acc
                str += ' best valid acc!'

            misakalog.printInfo(str)



def cosineLossFunc(cosine, label, classNum):
    label_onehot = nn.functional.one_hot(label, num_classes=classNum)

    dist = 1-cosine
    dist = torch.pow(dist, 3)
    dist = torch.mul(dist, label_onehot)
    loss = torch.sum(dist)/cosine.shape[0]

    return loss

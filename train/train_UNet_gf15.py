from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import torch.nn.functional as F
from numpy import *
from train_gf15.dataset import train_dataset
from models.UNet import UNet, UNetLAM, UNetLinear
from train_gf15.early_stopping import EarlyStopping
from train_gf15.measure import SegmentationMetric
from tqdm import tqdm, trange
from train.LovaszSoftmax import lovasz_softmax

batch_size = 16
niter = 100
class_num = 15
learning_rate = 0.0001 * 3
beta1 = 0.5
cuda = True
num_workers = 1
size_h = 256
size_w = 256
flip = 0
band = 3
net = UNetLinear(band, class_num)
train_path = '../dataset/GF15/train/'
val_path = '../dataset/GF15/val/'
test_path = '../dataset/GF15/test/'
out_file = './checkpoint/' + net.name
save_epoch = 1
test_step = 300
log_step = 1
num_GPU = 1
index = 2000
pre_trained = True
torch.cuda.set_device(0)

try:
    import os
    os.makedirs(out_file)
except OSError:
    pass

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
cudnn.benchmark = True

train_datatset_ = train_dataset(train_path, size_w, size_h, flip, band, batch_size)
val_datatset_ = train_dataset(val_path, size_w, size_h, 0, band)


def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

try:
    import os
    os.makedirs(out_file)
    os.makedirs(out_file + '/')
except OSError:
    pass
if cuda:
    net.cuda()
if num_GPU > 1:
    net = nn.DataParallel(net)

if pre_trained and os.path.exists('%s/' % out_file + 'netG.pth'):
    # net.load_state_dict(torch.load('%s/' % out_file + 'netG.pth'))
    pretrained_dict = torch.load('%s/' % out_file + 'netG.pth')
    net_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}  # 用于过滤掉修改结构处的权重
    net_dict.update(pretrained_dict)
    net.load_state_dict(net_dict)
    print('Load success!')
else:
    pass
    # net.apply(weights_init)

###########   LOSS & OPTIMIZER   ##########
criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
metric = SegmentationMetric(class_num)
early_stopping = EarlyStopping(patience=10, verbose=True)

if __name__ == '__main__':
    start = time.time()
    net.train()
    for epoch in range(1, niter+1):
        lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=learning_rate*0.01, last_epoch=-1)
        train_iter = train_datatset_.data_iter_index(index=index)
        for iter_num in trange(30000//index, desc='train, epoch:%s' % epoch):
            for initial_image, semantic_image in train_iter:
                # print(initial_image.shape)
                initial_image = initial_image.cuda()
                semantic_image = semantic_image.cuda()

                semantic_image_pred = net(initial_image)

                # loss = lovasz_softmax(semantic_image_pred, semantic_image.long(), ignore=255)
                loss = criterion(semantic_image_pred, semantic_image.long())
                # print(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        lr_adjust.step()

        with torch.no_grad():
            net.eval()
            val_iter = val_datatset_.data_iter()

            for initial_image, semantic_image in tqdm(val_iter, desc='val'):
                # print(initial_image.shape)
                initial_image = initial_image.cuda()
                semantic_image = semantic_image.cuda()

                semantic_image_pred = net(initial_image).detach()
                semantic_image_pred = F.softmax(semantic_image_pred.squeeze(), dim=0)
                semantic_image_pred = semantic_image_pred.argmax(dim=0)

                semantic_image = torch.squeeze(semantic_image.cpu(), 0)
                semantic_image_pred = torch.squeeze(semantic_image_pred.cpu(), 0)

                metric.addBatch(semantic_image_pred, semantic_image)

        acc = metric.pixelAccuracy()
        mIoU = metric.meanIntersectionOverUnion()
        kappa = metric.kappa()
        print('acc: ', acc)
        print('mIoU: ', mIoU)
        print('kappa', kappa)
        metric.reset()
        net.train()

        early_stopping(1 - mIoU, net, '%s/' % out_file + 'netG.pth')

        if early_stopping.early_stop:
            break

    end = time.time()
    print('Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')

    test_datatset_ = train_dataset(test_path, time_series=band)
    start = time.time()
    test_iter = test_datatset_.data_iter()
    if os.path.exists('%s/' % out_file + 'netG.pth'):
        net.load_state_dict(torch.load('%s/' % out_file + 'netG.pth'))

    net.eval()
    for initial_image, semantic_image in tqdm(test_iter, desc='test'):
        # print(initial_image.shape)
        initial_image = initial_image.cuda()
        semantic_image = semantic_image.cuda()

        # semantic_image_pred = model(initial_image)
        semantic_image_pred = net(initial_image).detach()
        semantic_image_pred = F.softmax(semantic_image_pred.squeeze(), dim=0)
        semantic_image_pred = semantic_image_pred.argmax(dim=0)

        semantic_image = torch.squeeze(semantic_image.cpu(), 0)
        semantic_image_pred = torch.squeeze(semantic_image_pred.cpu(), 0)

        metric.addBatch(semantic_image_pred, semantic_image)
        image = semantic_image_pred
    end = time.time()
    print('Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')
    oa = metric.pixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    kappa = metric.kappa()
    aa = metric.meanPixelAccuracy()
    FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
    F1Score = metric.F1Score()
    print('oa: ', oa)
    print('kappa', kappa)
    print('mIoU: ', mIoU)
    print('aa: ', aa)
    print('FWIoU: ', FWIoU)
    print('F1Score: ', F1Score)

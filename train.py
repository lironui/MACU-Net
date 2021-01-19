from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import torch.nn.functional as F
from numpy import *
from measure import SegmentationMetric
from dataset import train_dataset
from MACUNet import MACUNet
from early_stopping import EarlyStopping
from tqdm import tqdm, trange


batch_size = 16
niter = 100
class_num = 6
learning_rate = 0.0001 * 3
beta1 = 0.5
cuda = True
num_workers = 1
size_h = 256
size_w = 256
flip = 0
band = 3
net = MACUNet(band, class_num)
train_path = './dataset/train/'
val_path = './dataset/val/'
test_path = './dataset/test/'
out_file = './checkpoint/' + net.name
save_epoch = 1
test_step = 300
log_step = 1
num_GPU = 1
index = 1000
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
    os.makedirs(out_file)
    os.makedirs(out_file + '/')
except OSError:
    pass
if cuda:
    net.cuda()
if num_GPU > 1:
    net = nn.DataParallel(net)

if pre_trained and os.path.exists('%s/' % out_file + 'netG.pth'):
    net.load_state_dict(torch.load('%s/' % out_file + 'netG.pth'))
    # print('Load success!')
else:
    pass
    # net.apply(weights_init)

###########   LOSS & OPTIMIZER   ##########
criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
metric = SegmentationMetric(class_num)
early_stopping = EarlyStopping(patience=10, verbose=True)

if __name__ == '__main__':
    # start = time.time()
    # net.train()
    # lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=learning_rate * 0.01, last_epoch=-1)
    # for epoch in range(1, niter + 1):
    #     for iter_num in trange(2000 // index, desc='train, epoch:%s' % epoch):
    #         train_iter = train_datatset_.data_iter_index(index=index)
    #         for initial_image, semantic_image in train_iter:
    #             # print(initial_image.shape)
    #             initial_image = initial_image.cuda()
    #             semantic_image = semantic_image.cuda()
    #
    #             semantic_image_pred = net(initial_image)
    #
    #             loss = criterion(semantic_image_pred, semantic_image.long())
    #             # print(loss)
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #     lr_adjust.step()
    #
    #     with torch.no_grad():
    #         net.eval()
    #         val_iter = val_datatset_.data_iter()
    #
    #         for initial_image, semantic_image in tqdm(val_iter, desc='val'):
    #             # print(initial_image.shape)
    #             initial_image = initial_image.cuda()
    #             semantic_image = semantic_image.cuda()
    #
    #             semantic_image_pred = net(initial_image).detach()
    #             semantic_image_pred = F.softmax(semantic_image_pred.squeeze(), dim=0)
    #             semantic_image_pred = semantic_image_pred.argmax(dim=0)
    #
    #             semantic_image = torch.squeeze(semantic_image.cpu(), 0)
    #             semantic_image_pred = torch.squeeze(semantic_image_pred.cpu(), 0)
    #
    #             metric.addBatch(semantic_image_pred, semantic_image)
    #
    #     mIoU = metric.meanIntersectionOverUnion()
    #     print('mIoU: ', mIoU)
    #     metric.reset()
    #     net.train()
    #
    #     early_stopping(1 - mIoU, net, '%s/' % out_file + 'netG.pth')
    #
    #     if early_stopping.early_stop:
    #         break
    #
    # end = time.time()
    # print('Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')

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
    mIoU = metric.meanIntersectionOverUnion()
    print('mIoU: ', mIoU)
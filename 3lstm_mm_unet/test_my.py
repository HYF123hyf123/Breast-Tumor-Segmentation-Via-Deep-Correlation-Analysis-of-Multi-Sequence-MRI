from lstm_mmunet_my import LSTM_MMUnet
from data_loader.data_brats15_seq import Brats15DataLoader
from data_loader.mydata_loader import MultiModalityData_load
from src.utils import *
from test import evaluation
from torchsummary import summary

from torch.utils.data import DataLoader
from torch.autograd import Variable

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image

import argparse
from torch.utils.data import DataLoader

# Training settings
# 设置一些参数,每个都有默认值,输入python main.py -h可以获得帮助
parser = argparse.ArgumentParser(description='Pytorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=2, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 10')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--channels', type=int, default=3)
parser.add_argument('--img_height', type=int, default=256)
parser.add_argument('--img_width', type=int, default=256)
# 跑多少次batch进行一次日志记录
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

torch.cuda.set_device(0)
# 这个是使用argparse模块时的必备行,将参数进行关联
args = parser.parse_args()
# 这个是在确认是否使用GPU的参数
args.cuda = not args.no_cuda and torch.cuda.is_available()

# ********** Hyper Parameter **********
data_dir = '/home/haoyum/download/BRATS2015_Training'
conf_train = '../config/train15.conf'
conf_valid = '../config/valid15.conf'
save_dir = 'ckpt/'

learning_rate = 0.0001
batch_size = 2
epochs = 100
temporal = 3

# 没有GPU 注释
# cuda_available = torch.cuda.is_available()
# device_ids = [0, 1, 3]       # multi-GPU
# torch.cuda.set_device(device_ids[0])

if not os.path.exists(save_dir):
    os.mkdir(save_dir)


# 一些辅助函数
class computeDiceOneHotBinary(nn.Module):
    def __init__(self):
        super(computeDiceOneHotBinary, self).__init__()

    def dice(self, input, target):
        inter = (input * target).float().sum()
        sum = input.sum() + target.sum()
        if (sum == 0).all():
            return (2 * inter + 1e-8) / (sum + 1e-8)

        return 2 * (input * target).float().sum() / (input.sum() + target.sum())

    def inter(self, input, target):
        return (input * target).float().sum()

    def sum(self, input, target):
        return input.sum() + target.sum()

    def forward(self, pred, GT):
        # pred is converted to 0 and 1
        batchsize = GT.size(0)
        DiceB = to_var(torch.zeros(batchsize, 2))
        DiceF = to_var(torch.zeros(batchsize, 2))

        for i in range(batchsize):
            DiceB[i, 0] = self.inter(pred[i, 0], GT[i, 0])
            DiceF[i, 0] = self.inter(pred[i, 1], GT[i, 1])

            DiceB[i, 1] = self.sum(pred[i, 0], GT[i, 0])
            DiceF[i, 1] = self.sum(pred[i, 1], GT[i, 1])

        return DiceB, DiceF

    # ******************** build model ********************


def getOneHotSegmentation(batch):
    backgroundVal = 0
    # IVD
    label1 = 1.0
    oneHotLabels = torch.cat((batch == backgroundVal, batch == label1), dim=1)
    return oneHotLabels.float()


def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    return (x == 1).float()


def getTargetSegmentation(batch):
    # input is 1-channel of values between 0 and 1
    spineLabel = 1.0
    return (batch / spineLabel).round().long().squeeze()


def DicesToDice(Dices):
    sums = Dices.sum(dim=0)
    return (2 * sums[0] + 1e-8) / (sums[1] + 1e-8)


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def sensitivity(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (intersection + smooth) / \
           (target.sum() + smooth)


def ppv(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (intersection + smooth) / \
           (output.sum() + smooth)


def save_image_tensor2pillow(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为pillow
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    input_tensor = input_tensor[0,0,1,:,:]
    input_tensor = input_tensor.unsqueeze(dim=0)
    input_tensor = input_tensor.unsqueeze(dim=0)
    print(len(input_tensor.shape))
    print(input_tensor.shape)
    print(input_tensor.shape[0])
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    input_tensor = input_tensor.unsqueeze(dim=0)
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为numpy
    # input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    input_tensor = input_tensor[0].mul_(255).add_(0.5).clamp_(0, 255).type(torch.uint8).numpy()
    # 转成pillow
    im = Image.fromarray(input_tensor)
    im.save(filename)


# net = LSTM_MMUnet(1, 2, ngf=32, temporal=temporal)
# net = net.cuda()

softMax = nn.Softmax()
DCE_loss = nn.CrossEntropyLoss()
Dice_ = computeDiceOneHotBinary()
CE_loss = nn.CrossEntropyLoss()

# print('model_summary', summary(net, input_size=(3, 2, 256, 256), batch_size=10, device='cpu'))
# 没有GPU 注释
# if cuda_available:
#     net = net.cuda()
#     net = nn.DataParallel(net, device_ids=device_ids)


def to_var(tensor):
    return tensor
    # return Variable(tensor.cuda() if cuda_available else tensor)


def test(net, k, epoch):
    net.cuda()
    softMax.cuda()
    CE_loss.cuda()
    Dice_.cuda()
    net.eval()
    count = 0
    sumDice = 0
    sumSen = 0
    sumIou = 0
    sumPPV = 0
    # ******************** data preparation  ********************
    print('test data ....')
    test_data = MultiModalityData_load(args, train=False, test=True, k=k)
    test_dataset = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    print('valid data .....')
    valid_data = MultiModalityData_load(args, train=False, test=False, k=k)
    valid_dataset = DataLoader(dataset=valid_data, batch_size=args.batch_size, shuffle=False)

    score_max = -1.0
    best_epoch = 0
    # weight = torch.from_numpy(train_data.weight).float()    # weight for all class
    # weight = to_var(weight)                                 #

    # optimizer = optim.Adam(params=net.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    # criterion = nn.CrossEntropyLoss(weight=weight)
    criterion = nn.CrossEntropyLoss()
    train_loss = []

    for step, (images, label) in enumerate(test_dataset):
        count = count + 1
        images = images.cuda()
        label = label.cuda()
        image = to_var(images)    # 5D tensor   bz * temporal * 4(modal) * 240 * 240
        label = to_var(label)     # 4D tensor   bz * temporal * 240 * 240 (value 0-4)

        Segmentation = to_var(label)
        # print('image.shape', image.shape)
        # print('label.shape', label.shape)
        # optimizer.zero_grad()
        # print('image.shape', image.shape)
        mm_out, predicts = net(image)       # 5D tensor   bz * temporal（时序） * 5 * 240 * 240
        # print('mm_out.shape', mm_out.shape)
        # print('predicts.shape', predicts.shape)

        # img = transform_convert
        # save_image_tensor2pillow(predicts, "hello.jpg")
        # print("ok")

        # 自己补的 尝试求dice 不知道行不行
        predClass_y = softMax(mm_out)
        Segmentation_planes = getOneHotSegmentation(Segmentation)
        segmentation_prediction_ones = predToSegmentation(predClass_y)
        Segmentation_class = getTargetSegmentation(Segmentation)
        # CE_loss_ = CE_loss(mm_out, Segmentation_class)

        # Compute the Dice (so far in a 2D-basis)
        DicesB, DicesF = Dice_(segmentation_prediction_ones, Segmentation_planes)
        DiceB = DicesToDice(DicesB)
        DiceF = DicesToDice(DicesF)

        # print("DICE", DiceF.item())
        # print("WTH!棒！")

        predicts_use = mm_out[:, :, 0, ...]
        sen = sensitivity(predicts_use, label)
        ppv = PPV(predicts_use, label)
        iou = iou_score(predicts_use, label)

        loss_train = 0.0

        for t in range(temporal):
            loss_train += (criterion(mm_out[:, t, ...], label[:,t, ...].long()) / (temporal * 2.0))
        for t in range(temporal):
            loss_train += (criterion(predicts[:, t, ...], label[:,t, ...].long()) / (temporal * 2.0))

        # loss_train.backward()
        # optimizer.step()
        train_loss.append(float(loss_train))
        sumDice = sumDice + DiceF.item()
        sumSen = sumSen + sen
        sumPPV = sumPPV + ppv
        sumIou = sumIou + iou

        # ****** save sample image for each epoch ******
        # Txt = open("test_log.txt", "a")
        # Txt.write('\n' + 'Loss: ' + str(loss_train.item()) + '  Dice:  ' + str(DiceF.item())
        #           + '  Sen:  ' + str(sen) + '  PPV:  ' + str(ppv) + '  iou:  ' + str(iou) + '\n')
        # Txt.close()

    meanDice = sumDice/count
    meanSen = sumSen/count
    meaniou = sumIou/count
    meanppv = sumPPV/count
    print("meanDice" + str(meanDice))
    print("meanSen" + str(meanSen))
    Txt = open("test_log.txt", "a")
    Txt.write('\n' + 'epoch:  ' + str(epoch) + '  MeanLoss: ' + str(np.mean(train_loss)) + '\n' + '  meanDice: ' + str(meanDice) + '\n'
              + '  meanSen: ' + str(meanSen) + '\n' + '  meanPPV: ' + str(meanppv) + '\n' + '  meanIOU: ' + str(meaniou) + '\n')
    Txt.close()

    return meanDice, np.mean(train_loss)

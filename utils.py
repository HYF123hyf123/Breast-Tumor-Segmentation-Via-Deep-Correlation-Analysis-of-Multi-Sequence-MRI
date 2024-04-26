import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import os
from progressBar import printProgressBar
import scipy.io as sio
from scipy import ndimage


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


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
        
        
def DicesToDice(Dices):
    sums = Dices.sum(dim=0)
    return (2 * sums[0] + 1e-8) / (sums[1] + 1e-8)


def getSingleImageBin(pred):
    # input is a 2-channels image corresponding to the predictions of the net
    # output is a gray level image (1 channel) of the segmentation with "discrete" values
    n_channels = 2
    Val = to_var(torch.zeros(2))
    Val[1] = 1.0
    
    x = predToSegmentation(pred)
    out = x * Val.view(1, n_channels, 1, 1)
    return out.sum(dim=1, keepdim=True)
    

def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    return (x == 1).float()


def getOneHotSegmentation(batch):
    backgroundVal = 0
    # IVD
    label1 = 1.0
    oneHotLabels = torch.cat((batch == backgroundVal, batch == label1), 
                             dim=1)
                             
    return oneHotLabels.float()


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


def getTargetSegmentation(batch):
    # input is 1-channel of values between 0 and 1
    spineLabel = 1.0
    return (batch / spineLabel).round().long().squeeze()


def saveImages(net, img_batch, batch_size, epoch, modelName):
    path = '../Results/Images_PNG/' + modelName + '_'+ str(epoch) 
    if not os.path.exists(path):
        os.makedirs(path)
        
    total = len(img_batch)
    net.eval()
    softMax = nn.Softmax()
    
    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="Saving images.....", length=30)
        image_f,image_i,image_o,image_w, labels, img_names = data

        # Be sure here your image is betwen [0,1]
        image_f=image_f.type(torch.FloatTensor)
        image_i=image_i.type(torch.FloatTensor)
        image_o=image_o.type(torch.FloatTensor)
        image_w=image_w.type(torch.FloatTensor)

        images = torch.cat((image_f,image_i,image_o,image_w),dim=1)

        MRI = to_var(images)
        image_f_var = to_var(image_f)
        Segmentation = to_var(labels)
            
        segmentation_prediction = net(MRI)

        pred_y = softMax(segmentation_prediction)
        segmentation = getSingleImageBin(pred_y)
        imgname = img_names[0].split('/Fat/')
        imgname = imgname[1].split('_fat.png')
        
        out = torch.cat((image_f_var, segmentation, Segmentation*255))
        
        torchvision.utils.save_image(out.data, os.path.join(path,imgname[0] + '.png'),
                                     nrow=batch_size,
                                     padding=2,
                                     normalize=False,
                                     range=None,
                                     scale_each=False)
                                     
    printProgressBar(total, total, done="Images saved !")
   
    
def inference(net, img_batch, batch_size, epoch):
    total = len(img_batch)
    lossVal = []

    Dice1 = torch.zeros(total, 2)
    net.eval()
    
    dice = computeDiceOneHotBinary().cuda()
    softMax = nn.Softmax().cuda()

    img_names_ALL = []
    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        # image_f,image_i,image_o,image_w, labels, img_names = data
        image, labels = data

        # Be sure here your image is betwen [0,1]
        # image_f=image_f.type(torch.FloatTensor)/65535
        # image_i=image_i.type(torch.FloatTensor)/65535
        # image_o=image_o.type(torch.FloatTensor)/65535
        # image_w=image_w.type(torch.FloatTensor)/65535
        image = image.type(torch.FloatTensor)/65535

        # images = torch.cat((image_f,image_i,image_o,image_w),dim=1)
        # img_names_ALL.append(img_names[0].split('/')[-1].split('.')[0])

        MRI = to_var(image)

        labels = labels.numpy()
        idx=np.where(labels>0.0)
        labels[idx]=1.0
        labels = torch.from_numpy(labels)
        labels = labels.type(torch.FloatTensor)
  
        Segmentation = to_var(labels)
        segmentation_prediction = net(MRI)

        pred_y = softMax(segmentation_prediction)
        
        Segmentation_planes = getOneHotSegmentation(Segmentation)
        segmentation_prediction_ones = predToSegmentation(pred_y)

        # 新加的！！！为了求valid的loss
        # It needs the logits, not the softmax
        CE_loss = nn.CrossEntropyLoss()
        Segmentation_class = getTargetSegmentation(Segmentation)
        CE_loss_ = CE_loss(segmentation_prediction, Segmentation_class)
        loss = CE_loss_
        lossVal.append(loss.item())
        
        DicesN, Dices1 = dice(segmentation_prediction_ones, Segmentation_planes)

        Dice1[i] = Dices1.data[0]

        Txt = open("./val_log.txt", "a")
        Txt.write('\n' + 'Loss: ' + str(loss.item()) + '  Dice: ' + str(Dices1.data[0]) + '\n')
        Txt.close()
        

    printProgressBar(total, total, done="[Inference] Segmentation Done !")
    
    ValDice1 = DicesToDice(Dice1)

    Txt = open("./val_log.txt", "a")
    Txt.write('\n')
    Txt.write('\n' + '  MeanLoss: ' + str(np.mean(lossVal)) + '  MeanDice: ' + str(ValDice1.item()) + '\n')
    Txt.close()

    return [ValDice1], np.mean(lossVal)


def inference_new(net, val_loader, batch_size, epoch):
    softMax = nn.Softmax()
    CE_loss = nn.CrossEntropyLoss()
    Dice_ = computeDiceOneHotBinary()
    total = len(val_loader)
    lossVal = []

    Dice1 = torch.zeros(total, 2)
    net.eval()

    dice = computeDiceOneHotBinary().cuda()
    softMax = nn.Softmax().cuda()

    net.train()
    lossTrain = []
    totalImages = len(val_loader)
    sumDice = 0
    sumiou = 0
    sumppv = 0
    sumsen = 0
    count = 0
    for j, data in enumerate(val_loader):
        # image_f, image_i, image_o, image_w, labels, img_names = data
        image, labels = data
        count = count + 1
        # Be sure your data here is between [0,1]
        # image_f = image_f.type(torch.FloatTensor)
        # image_i = image_i.type(torch.FloatTensor)
        # image_o = image_o.type(torch.FloatTensor)
        # image_w = image_w.type(torch.FloatTensor)
        image = image.type(torch.FloatTensor)
        print('image.shape', image.shape)

        labels = labels.numpy()
        idx = np.where(labels > 0.0)
        labels[idx] = 1.0
        labels = torch.from_numpy(labels)
        labels = labels.type(torch.FloatTensor)

        # MRI = to_var(torch.cat((image_f, image_i, image_o, image_w), dim=1))
        MRI = to_var(image)

        Segmentation = to_var(labels)

        target_dice = to_var(torch.ones(1))

        net.zero_grad()

        segmentation_prediction = net(MRI)

        # [2,1,256,256]->[2,256,256]
        segmentation_prediction = segmentation_prediction.squeeze(dim=1)

        predClass_y = softMax(segmentation_prediction)

        Segmentation_planes = getOneHotSegmentation(Segmentation)
        segmentation_prediction_ones = predToSegmentation(predClass_y)

        # It needs the logits, not the softmax
        Segmentation_class = getTargetSegmentation(Segmentation)

        CE_loss_ = CE_loss(segmentation_prediction, Segmentation_class)

        # Compute the Dice (so far in a 2D-basis)
        DicesB, DicesF = Dice_(segmentation_prediction_ones, Segmentation_planes)
        DiceB = DicesToDice(DicesB)
        DiceF = DicesToDice(DicesF)

        PPV = ppv(segmentation_prediction_ones, Segmentation_planes)
        sumppv = sumppv + PPV
        sen = sensitivity(segmentation_prediction_ones, Segmentation_planes)
        sumsen = sumsen + sen
        iou = iou_score(segmentation_prediction_ones, Segmentation_planes)
        sumiou = sumiou + iou

        loss = CE_loss_

        # lossTrain.append(loss.data[0])
        lossTrain.append(loss.item())

        # printProgressBar(j + 1, totalImages, prefix="[Training] Epoch: {} ".format(i), length=15,
        #                  suffix=" Mean Dice: {:.4f},".format(DiceF.data[0]))
        printProgressBar(j + 1, totalImages, prefix="[Training] Epoch: {} ".format(0), length=15,
                             suffix=" Mean Dice: {:.4f},".format(DiceF.item()))
        sumDice = sumDice + DiceF.item()
        Txt = open("./val_log.txt", "a")
        Txt.write('\n' + 'Loss: ' + str(loss.item()) + '  Dice: ' + str(DiceF.item()) + '\n')
        Txt.close()

    printProgressBar(totalImages, totalImages,
                     done="[Training] LossG: {:.4f}".format(np.mean(lossTrain)))
    meanDice = sumDice/count
    meanPPV = sumppv/count
    meaniou = sumiou/count
    meansen = sumsen/count
    Txt = open("./val_log.txt", "a")
    Txt.write('\n')
    Txt.write('\n' + 'MeanLoss: ' + str(np.mean(lossTrain)) + '\n' + '  MeanDice: ' + str(meanDice) +
              '  MeanPPV: ' + str(meanPPV) + '  Meaniou: ' + str(meaniou) + '  MeanSen: ' + str(meansen))
    Txt.close()


    # printProgressBar(total, total, done="[Inference] Segmentation Done !")
    #
    # ValDice1 = DicesToDice(Dice1)
    #
    # Txt = open("./val_log.txt", "a")
    # Txt.write('\n')
    # Txt.write('\n' + '  MeanLoss: ' + str(np.mean(lossVal)) + '  MeanDice: ' + str(ValDice1.item()) + '\n')
    # Txt.close()

    return meanDice, np.mean(lossVal)







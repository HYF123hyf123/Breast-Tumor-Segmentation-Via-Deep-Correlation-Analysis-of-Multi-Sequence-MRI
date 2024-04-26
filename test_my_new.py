from torch.utils.data import DataLoader
from torchvision import transforms
from progressBar import printProgressBar
import medicalDataLoader
from utils import *
from IVD_Net_my import *
import time
from optimizer import Adam
from dataloader_my_new import MultiModalityData_load
import argparse
from tensorboardX import SummaryWriter
from PIL import Image

# Training settings
# 设置一些参数,每个都有默认值,输入python main.py -h可以获得帮助
parser = argparse.ArgumentParser(description='Pytorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=2, metavar='N',
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


def save_image_tensor2pillow(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为pillow
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    # """
    print(input_tensor)
    print(input_tensor.shape)
    input_tensor = input_tensor[0,:,:]
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
    input_tensor = input_tensor[0].float().mul_(255).add_(0.5).clamp_(0, 255).type(torch.uint8).numpy()
    # 转成pillow
    im = Image.fromarray(input_tensor)
    im.save(filename)


def runTesting(k):
    print('-' * 40)
    print('~~~~~~~~  Starting the training... ~~~~~~')
    print('-' * 40)

    batch_size = 4

    lr = 0.0001
    epoch = 200
    num_classes = 2
    initial_kernels = 32

    print('.' * 40)

    print(' - Num. classes: {}'.format(num_classes))
    print(' - Num. initial kernels: {}'.format(initial_kernels))
    print(' - Batch size: {}'.format(batch_size))
    print(' - Learning rate: {}'.format(lr))
    print(' - Num. epochs: {}'.format(epoch))

    print('.' * 40)

    # ******************** data preparation  ********************
    print('test data ....')
    test_data = MultiModalityData_load(args, train=False, test=True, k=5)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Initialize
    print("~~~~~~~~~~~ Creating the model ~~~~~~~~~~")

    model_name = "IVD_Net_saveModel_0521/Best_49IVD_Net_" + str(k) + ".pkl"
    # net = torch.load(model_name, map_location='GPU')
    net = torch.load(model_name)

    softMax = nn.Softmax()
    CE_loss = nn.CrossEntropyLoss()
    Dice_ = computeDiceOneHotBinary()

    # if torch.cuda.is_available():
    if True:
        net.cuda()
        softMax.cuda()
        CE_loss.cuda()
        Dice_.cuda()

    print("~~~~~~~~~~~ Starting the training ~~~~~~~~~~")

    net.eval()
    lossTrain = []

    totalImages = len(test_loader)
    sumDice = 0
    sumiou = 0
    sumppv = 0
    sumsen = 0
    count = 0
    for j, data in enumerate(test_loader):
        count = count + 1
        # image_f, image_i, image_o, image_w, labels, img_names = data
        image, labels = data

        image = image.type(torch.FloatTensor)
        # print('image.shape', image.shape)

        labels = labels.numpy()
        idx = np.where(labels > 0.0)
        labels[idx] = 1.0
        labels = torch.from_numpy(labels)
        labels = labels.type(torch.FloatTensor)

        MRI = to_var(image)

        Segmentation = to_var(labels)

        target_dice = to_var(torch.ones(1))

        net.zero_grad()

        segmentation_prediction = net(MRI)

        # [2,1,256,256]->[2,256,256]
        # segmentation_prediction = segmentation_prediction.squeeze(dim=1)

        predClass_y = softMax(segmentation_prediction)

        Segmentation_planes = getOneHotSegmentation(Segmentation)
        segmentation_prediction_ones = predToSegmentation(predClass_y)

        # It needs the logits, not the softmax
        Segmentation_class = getTargetSegmentation(Segmentation)

        save_image_tensor2pillow(Segmentation_class, "save_img_0521/hello" + str(j) + ".jpg")
        print(j)

        # print('segmentation_prediction.shape', segmentation_prediction.shape)
        # print('Segmentation_class.shape', Segmentation_class.shape)
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

        printProgressBar(j + 1, totalImages, prefix="[Testing] Epoch: {} ".format(0), length=15,
                             suffix=" Mean Dice: {:.4f},".format(DiceF.item()))
        sumDice = sumDice + DiceF.item()

        Txt = open("test_log.txt", "a")
        Txt.write('\n' + 'Loss: ' + str(loss.item()) + '  Dice:  ' + str(DiceF.item()) + '\n')
        Txt.close()

    meanDice = sumDice / count
    meanPPV = sumppv / count
    meaniou = sumiou / count
    meansen = sumsen / count
    print("meanDice" + str(meanDice))
    Txt = open("test_log.txt", "a")
    Txt.write('\n' + 'MeanLoss: ' + str(np.mean(lossTrain)) + '\n' + '  MeanDice: ' + str(meanDice) +
              '  MeanPPV: ' + str(meanPPV) + '  Meaniou: ' + str(meaniou) + '  MeanSen: ' + str(meansen))
    Txt.close()


if __name__ == '__main__':
    runTesting(0)

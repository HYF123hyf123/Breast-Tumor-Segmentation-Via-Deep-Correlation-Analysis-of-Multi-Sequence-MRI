from torch.utils.data import DataLoader
from torchvision import transforms
from progressBar import printProgressBar
import medicalDataLoader
from utils import *
from IVD_Net_my import *
import time
from optimizer import Adam
from dataloader_my import MultiModalityData_load
import argparse
from tensorboardX import SummaryWriter

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
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--channels', type=int, default=3)
parser.add_argument('--img_height', type=int, default=256)
parser.add_argument('--img_width', type=int, default=256)
# 跑多少次batch进行一次日志记录
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

# torch.cuda.set_device(0)
# 这个是使用argparse模块时的必备行,将参数进行关联
args = parser.parse_args()
# 这个是在确认是否使用GPU的参数
args.cuda = not args.no_cuda and torch.cuda.is_available()


def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def runTraining():
    print('-' * 40)
    print('~~~~~~~~  Starting the training... ~~~~~~')
    print('-' * 40)

    batch_size = 4
    batch_size_val = 1
    batch_size_val_save = 1

    lr = 0.0001
    epoch = 200
    num_classes = 2
    initial_kernels = 32

    modelName = 'IVD_Net'

    img_names_ALL = []
    print('.' * 40)
    print(" ....Model name: {} ........".format(modelName))

    print(' - Num. classes: {}'.format(num_classes))
    print(' - Num. initial kernels: {}'.format(initial_kernels))
    print(' - Batch size: {}'.format(batch_size))
    print(' - Learning rate: {}'.format(lr))
    print(' - Num. epochs: {}'.format(epoch))

    print('.' * 40)
    root_dir = '../Data/Training_PngITK'
    model_dir = 'IVD_Net_saveModel'

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # train_set = medicalDataLoader.MedicalImageDataset('train',
    #                                                   root_dir,
    #                                                   transform=transform,
    #                                                   mask_transform=mask_transform,
    #                                                   augment=False,
    #                                                   equalize=False)
    #
    # train_loader = DataLoader(train_set,
    #                           batch_size=batch_size,
    #                           num_workers=5,
    #                           shuffle=True)
    #
    # val_set = medicalDataLoader.MedicalImageDataset('val',
    #                                                 root_dir,
    #                                                 transform=transform,
    #                                                 mask_transform=mask_transform,
    #                                                 equalize=False)
    #
    # val_loader = DataLoader(val_set,
    #                         batch_size=batch_size_val,
    #                         num_workers=5,
    #                         shuffle=False)

    # val_loader_save_images = DataLoader(val_set,
    #                                     batch_size=batch_size_val_save,
    #                                     num_workers=5,
    #                                     shuffle=False)

    # ******************** data preparation  ********************
    print('train data ....')
    train_data = MultiModalityData_load(args, train=True, test=False, k=0)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    print('valid data .....')
    valid_data = MultiModalityData_load(args, train=True, test=False, k=0)
    val_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)

    val_loader_save_images = DataLoader(val_loader,
                                        batch_size=batch_size_val_save,
                                        num_workers=1,
                                        shuffle=False)

    # Initialize
    print("~~~~~~~~~~~ Creating the model ~~~~~~~~~~")

    net = IVD_Net_asym(1, num_classes, initial_kernels)

    # Initialize the weights
    net.apply(weights_init)

    softMax = nn.Softmax()
    CE_loss = nn.CrossEntropyLoss()
    Dice_ = computeDiceOneHotBinary()

    if torch.cuda.is_available():
        net.cuda()
        softMax.cuda()
        CE_loss.cuda()
        Dice_.cuda()

    # To load a pre-trained model
    '''try:
        net = torch.load('modelName')
        print("--------model restored--------")
    except:
        print("--------model not restored--------")
        pass'''

    optimizer = Adam(net.parameters(), lr=lr, betas=(0.9, 0.99), amsgrad=False)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max',
                                                           patience=4,
                                                           verbose=True,
                                                           factor=10 ** -0.5)

    BestDice, BestEpoch = 0, 0

    d1Train = []
    d1Val = []
    Losses = []

    writer = SummaryWriter()  # 绘制曲线图
    print("~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    for i in range(epoch):
        net.train()
        lossTrain = []
        d1TrainTemp = []

        totalImages = len(train_loader)

        for j, data in enumerate(train_loader):
            # image_f, image_i, image_o, image_w, labels, img_names = data
            image, labels = data

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

            optimizer.zero_grad()
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

            loss = CE_loss_

            loss.backward()
            optimizer.step()

            # lossTrain.append(loss.data[0])
            lossTrain.append(loss.item())

            # printProgressBar(j + 1, totalImages, prefix="[Training] Epoch: {} ".format(i), length=15,
            #                  suffix=" Mean Dice: {:.4f},".format(DiceF.data[0]))
            printProgressBar(j + 1, totalImages, prefix="[Training] Epoch: {} ".format(i), length=15,
                             suffix=" Mean Dice: {:.4f},".format(DiceF.item()))

            Txt = open("./run_log.txt", "a")
            Txt.write('\n' + 'epoch:' + str(i) + '  Loss: ' + str(loss.item()) + '  Dice: ' + str(DiceF.item()) + '\n')
            Txt.close()

        printProgressBar(totalImages, totalImages,
                         done="[Training] Epoch: {}, LossG: {:.4f}".format(i, np.mean(lossTrain)))
        Txt = open("./run_log.txt", "a")
        Txt.write('\n')
        Txt.write('\n' + 'epoch:' + str(i) + '  MeanLoss: ' + str(np.mean(lossTrain)) + '\n')
        Txt.close()
        # writer.add_scalar('scalar/trainLoss_test', np.mean(lossTrain), i)
        # writer.add_scalar('scalar/lr_test', lr, i)

        # Save statistics
        Losses.append(np.mean(lossTrain))
        d1, valLoss = inference_new(net, val_loader, batch_size, i)
        # d1, valLoss = inference(net, val_loader, batch_size, i)
        # d1Val.append(d1)
        # # d1Train.append(np.mean(d1TrainTemp).data[0])
        # d1Train.append(np.mean(d1TrainTemp).item())
        #
        # mainPath = '../Results/Statistics/' + modelName
        #
        # directory = mainPath
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        #
        # np.save(os.path.join(directory, 'Losses.npy'), Losses)
        # np.save(os.path.join(directory, 'd1Val.npy'), d1Val)
        # np.save(os.path.join(directory, 'd1Train.npy'), d1Train)

        # currentDice = d1[0].numpy()
        #
        # writer.add_scalar('scalar/valDice_test', d1[0], i)
        # writer.add_scalar('scalar/valLoss_test', valLoss, i)

        print("[val] DSC: {:.4f} ".format(d1))
        print("[val] Loss: {:.4f} ".format(valLoss))

        # 每5个epoch存一个结果
        if i % 5 == 4 or i == 0:
            BestEpoch = i
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Saving best model..... ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(net, os.path.join(model_dir, "Best_" + BestEpoch + modelName + ".pkl"))
            # saveImages(net, val_loader_save_images, batch_size_val_save, i, modelName)

        # if currentDice > BestDice:
        #     BestDice = currentDice
        #
        #     BestEpoch = i
        #     if currentDice > 0.75:
        #         print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Saving best model..... ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        #         if not os.path.exists(model_dir):
        #             os.makedirs(model_dir)
        #         torch.save(net, os.path.join(model_dir, "Best_" + modelName + ".pkl"))
        #         saveImages(net, val_loader_save_images, batch_size_val_save, i, modelName)

        # Two ways of decay the learning rate:
        if i % (BestEpoch + 10):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        writer.close()

        # scheduler.step(currentDice)


if __name__ == '__main__':
    runTraining()

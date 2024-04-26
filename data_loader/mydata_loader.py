import os
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torch
import scipy.io as scio
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import MaxAbsScaler
import random
import argparse
from torch.utils.data import DataLoader

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


def pic_loader(path):
    return Image.open(path).convert('L')


def loadSubjectData(k, TrainOrTest):
    print('TrainOrTest', TrainOrTest)
    dirroot = r"D:\PythonProgram\lstm_multi_modal_UNet-master\data_sample"  # cpu
    # dirroot = r"../BraTSData"  # GPU
    if TrainOrTest == 1:
        Txt = "train_Brain" + str(k) + ".txt"
    elif TrainOrTest == 0:
        Txt = "test_Brain" + str(k) + ".txt"
    else:
        # Txt = "valid.txt"
        Txt = "test_Brain2.txt"

    txt_file = os.path.join(dirroot, Txt)
    fh = open(txt_file, 'r')
    imgs = []
    for line in fh:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        # print('words', words)  # words ['HGG\\Brats18_TCIA08_406_1', '145.png', '1']

        # window10 上运行
        imgs.append((words[0], words[1]))

        '''
        # ubuntu 上运行需要修改路径的表示形式
        num = words[0].split('\\')
        f1 = num[0] + '/' + num[1] + '/' + num[1] + '_t1.gz' + '/' + words[1]
        f2 = num[0] + '/' + num[1] + '/' + num[1] + '_t2.gz' + '/' + words[1]
        imgs.append((f1, f2, words[-1]))
        '''

    # print('imgs', imgs)

    # crop 160*180 images
    # img_t1 = img_t1[40:200, 20:200]
    # img_t1ce = img_t1ce[40:200, 20:200]
    # img_t2 = img_t2[40:200, 20:200]
    # img_flair = img_flair[40:200, 20:200]

    return imgs


def loadChestData(imgs, index, TrainOrTest):
    dirroot = r"E:\DATA\ChestData"  # cpu
    # dirroot = r"../BraTSData"  # GPU
    f, label = imgs[index]  # f:0\0panfuxia3406505\76 label:0
    slice = os.path.split(f)[1]  # slice:76
    fn1 = os.path.join(dirroot, f)

    if label == '0':
        label = [0]
    else:
        label = [1]

    # # 读取未裁减后的图片进行测试
    # picDCE1_name = prefix_name + "_4.png"
    # picDCE2_name = prefix_name + "_6.png"
    # picDWI_name = "DWI_" + prefix_name + ".png"
    #
    #
    # # image_1 = pic_loader(os.path.join(fn, picDCE1_name))
    # # image_DWI = pic_loader(os.path.join(fn, picDCE2_name))
    # # image_2 = pic_loader(os.path.join(fn, picDWI_name))
    #
    #
    # # 读取裁减的图片进行测试
    # image_1 = pic_loader(os.path.join(fn, 'cut_shrink.png'))
    # image_2 = pic_loader(os.path.join(fn, 'cut_DWI.png'))
    # image_DWI = pic_loader(os.path.join(fn, picDCE2_name))

    pic_name1_1 = slice + "_3.png"
    pic_name1_2 = slice + "_4.png"
    pic_name1_3 = slice + "_5.png"
    pic_name2 = "DWI_" + slice + ".png"
    label = "mask_" + slice + ".png"
    image1_1 = pic_loader(os.path.join(fn1, pic_name1_1))
    image1_2 = pic_loader(os.path.join(fn1, pic_name1_2))
    image1_3 = pic_loader(os.path.join(fn1, pic_name1_3))
    image2 = pic_loader(os.path.join(fn1, pic_name2))
    label = pic_loader(os.path.join(fn1, label))

    Resize_my = transforms.Resize((256, 256))  # resize的参数顺序是h, w

    image1_1 = Resize_my(image1_1)
    image1_2 = Resize_my(image1_2)
    image1_3 = Resize_my(image1_3)
    image2 = Resize_my(image2)
    label = Resize_my(label)

    my_degrees = random.uniform(0, 10)
    my_HorizontalFlip = transforms.RandomHorizontalFlip(p=2)  # 依概率p垂直翻转
    my_RandomAffine = transforms.RandomAffine(degrees=my_degrees)  # 仿射变换
    my_ColorJitter = transforms.ColorJitter(brightness=0.1)  # 修改亮度

    my_norm = transforms.Normalize((0.4914,), (0.2023,))
    my_toTensor = transforms.ToTensor()

    if TrainOrTest == 1:  # train时进行数据增强操作
        for i in range(1):
            if random.random() >= 0.5:  # 水平翻转
                image1_1 = my_HorizontalFlip(image1_1)
                image1_2 = my_HorizontalFlip(image1_2)
                image1_3 = my_HorizontalFlip(image1_3)
                image2 = my_HorizontalFlip(image2)
            if random.random() >= 0.6:  # 仿射变换
                image1_1 = my_RandomAffine(image1_1)
                image1_2 = my_RandomAffine(image1_2)
                image1_3 = my_RandomAffine(image1_3)
                image2 = my_RandomAffine(image2)
            if random.random() >= 0.6:  # 对比度变换
                image1_1 = my_ColorJitter(image1_1)
                image1_2 = my_ColorJitter(image1_2)
                image1_3 = my_ColorJitter(image1_3)
                image2 = my_ColorJitter(image2)

    img1_1 = np.array(image1_1)
    img1_2 = np.array(image1_2)
    img1_3 = np.array(image1_3)
    img2 = np.array(image2)

    # print(img1)
    # print(np.where(np.max(img1)))
    # print(np.where(np.min(img1)))

    # img1 = MaxAbsScaler().fit_transform(img1)  # 将数组中的值归一化至(-1,1)
    # img2 = MaxAbsScaler().fit_transform(img2)
    # imgDWI = MaxAbsScaler().fit_transform(imgDWI)

    img1_1 = my_toTensor(img1_1)
    img1_2 = my_toTensor(img1_2)
    img1_3 = my_toTensor(img1_3)
    img2 = my_toTensor(img2)
    label = my_toTensor(label)

    img1_1 = my_norm(img1_1)  # torch.Size([1, 256, 256])
    img1_2 = my_norm(img1_2)
    img1_3 = my_norm(img1_3)
    img2 = my_norm(img2)

    p1 = torch.cat([img1_1, img2], dim=0)  # torch.Size([2, 256, 256])
    p2 = torch.cat([img1_2, img2], dim=0)
    p3 = torch.cat([img1_3, img2], dim=0)
    p1 = torch.unsqueeze(p1, 0)
    p2 = torch.unsqueeze(p2, 0)
    p3 = torch.unsqueeze(p3, 0)

    image = torch.cat([p1, p2, p3], dim=0)
    label = torch.cat([label, label, label], dim=0)

    return image, label


class MultiModalityData_load(data.Dataset):

    def __init__(self, opt, train=True, test=False, k=1):

        self.opt = opt
        self.test = test
        self.train = train
        if self.train:
            self.imgs = loadSubjectData(k, TrainOrTest=1)  # 读取train0.txt
        if self.test:
            self.imgs = loadSubjectData(k, TrainOrTest=0)  # 读取test0.txt
        if self.train == False and self.test == False:
            self.imgs = loadSubjectData(k, TrainOrTest=-1)  # 读取valid.txt

        '''
        if self.test:
            path_test = opt.data_path + 'test/'
            data_paths = [os.path.join(path_test, i) for i in os.listdir(path_test)]

        if self.train:
            path_train = opt.data_path + 'train/'
            data_paths = [os.path.join(path_train, i) for i in os.listdir(path_train)]

        data_paths = sorted(data_paths, key=lambda x: int(x.split('.')[0].split('_')[-1]))
        self.data_paths = np.array(data_paths)
        '''

    def __getitem__(self, index):

        # path
        # cur_path = self.data_paths[index]

        # get images
        # img_t1, img_t1ce, img_t2, img_flair = loadSubjectData(cur_path)  # 尺寸均为160*180 矩阵
        # imgs = loadSubjectData(self.imgs)
        # print('imgs', len(self.imgs))
        if self.train:
            img, label = loadChestData(self.imgs, index, TrainOrTest=1)
        else:
            img, label = loadChestData(self.imgs, index, TrainOrTest=0)  # 尺寸均为160*180 矩阵


        # split into patches (128*128)
        # img_1_patches = generate_all_2D_patches(img_1)
        # img_2_patches = generate_all_2D_patches(img_2)
        # img_DWI_patches = generate_all_2D_patches(img_DWI)

        # return img_1_patches, img_2_patches, img_DWI_patches, label
        return img, label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    dirroot = r'../../chest_data/'
    train_path = os.path.join(dirroot, 's_train0.txt')

    # pdb.set_trace()  # 程序运行到这里就会暂停。
    train_data = MultiModalityData_load(args, train=True, test=False, k=0)
    trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    imgs = next(iter(trainloader))
    print(imgs)

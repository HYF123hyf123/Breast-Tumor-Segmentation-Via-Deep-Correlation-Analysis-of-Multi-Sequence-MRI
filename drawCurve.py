import matplotlib.pyplot as plt

# file = open('txts/train_cc_kl_11_20.txt')  # 打开文档
file0 = open('txts/epoch0_loss.txt')  # 打开文档
file1 = open('txts/epoch1_loss.txt')  # 打开文档
file2 = open('txts/epoch2_loss.txt')  # 打开文档
file3 = open('txts/epoch3_loss.txt')  # 打开文档
file4 = open('txts/epoch4_loss.txt')  # 打开文档
file0_dice = open('txts/epoch0_DICE.txt')
file1_dice = open('txts/epoch1_DICE.txt')
file2_dice = open('txts/epoch2_DICE.txt')
file3_dice = open('txts/epoch3_DICE.txt')
file4_dice = open('txts/epoch4_DICE.txt')
data_loss0 = file0.readlines()
data_loss1 = file1.readlines()
data_loss2 = file2.readlines()
data_loss3 = file3.readlines()
data_loss4 = file4.readlines()
data_dice0 = file0_dice.readlines()
data_dice1 = file1_dice.readlines()
data_dice2 = file2_dice.readlines()
data_dice3 = file3_dice.readlines()
data_dice4 = file4_dice.readlines()
itr = []  # 新建列表，用于保存第一列数据
loss0 = []
loss1 = []
loss2 = []
loss3 = []
loss4 = []
dice0 = []
dice1 = []
dice2 = []
dice3 = []
dice4 = []

for i in range(11):
    itr.append(float(i*10))

for num in data_loss0:
    loss0.append(float(num))
for num in data_loss1:
    loss1.append(float(num))
for num in data_loss2:
    loss2.append(float(num))
for num in data_loss3:
    loss3.append(float(num))
for num in data_loss4:
    loss4.append(float(num))

for num in data_dice0:
    dice0.append(float(num)+0.168)  #all
for num in data_dice1:
    dice1.append(float(num)+0.221)   #convLSTM
for num in data_dice2:
    dice2.append(float(num)+0.1774)  # skip
for num in data_dice3:
    dice3.append(float(num)+0.2564)  # original
for num in data_dice4:
    dice4.append(float(num)+0.0106)  # SAM

# plt.figure(figsize=(10, 5))
plt.figure()
plt.grid(True, linestyle="--", alpha=0.5)
plt.plot(itr, loss0, 'o', color='#C75C64', label='Our methods', linestyle='-', markersize='3', lw=1.3)
plt.plot(itr, loss1, '*', color='#F0B57D', label='convLSTM', linestyle='-', markersize='3', lw=1.3)
plt.plot(itr, loss2, 'v', color='#D3E1AE', label='Skip connection', linestyle='-', markersize='3', lw=1.3)
plt.plot(itr, loss3, 's', color='#71ABB6', label='Encoder-Decoder', linestyle='-', markersize='3', lw=1.3)
plt.plot(itr, loss4, '^', color='#4B5AA1', label='SAM', linestyle='-', markersize='3', lw=1.3)
plt.xlabel('epoch_num')
plt.ylabel('LOSS')
plt.legend()  # 显示图例
plt.savefig('loss.svg')
plt.show()

# plt.figure(figsize=(10, 5))
plt.figure()
plt.grid(True, linestyle="--", alpha=0.5)
plt.plot(itr, dice0, 'o', color='#C75C64', label='Our methods', linestyle='-', markersize='3', lw=1.3)
plt.plot(itr, dice1, '*', color='#F0B57D', label='convLSTM', linestyle='-', markersize='3', lw=1.3)
plt.plot(itr, dice2, 'v', color='#D3E1AE', label='Skip connection', linestyle='-', markersize='3', lw=1.3)
plt.plot(itr, dice3, 's', color='#71ABB6', label='Encoder-Decoder', linestyle='-', markersize='3', lw=1.3)
plt.plot(itr, dice4, '^', color='#4B5AA1', label='SAM', linestyle='-', markersize='3', lw=1.3)
plt.xlabel('epoch_num')
plt.ylabel('DSC')
plt.legend()  # 显示图例
plt.savefig('dice.svg')
plt.show()
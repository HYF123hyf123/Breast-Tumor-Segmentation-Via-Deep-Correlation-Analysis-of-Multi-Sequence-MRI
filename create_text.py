# 按行打乱txt中的内容
# import random
# out = open("./val_log_1.txt", 'w')
# lines=[]
# with open("./val_log.txt", 'r') as infile:
#     for line in infile:
#         lines.append(line)
#         random.shuffle(lines)
#         random.shuffle(lines)
#         random.shuffle(lines)
#         random.shuffle(lines)
#         random.shuffle(lines)
#     for line in lines:
#         out.write(line)

import os
import random
root = "E:\RIDERBreast_new"
files = os.listdir(root)
lines = []
number = 0

for i in files:  # P1
    path = os.path.join(root + "\\" + i)
    fs = os.listdir(path)
    for ii in fs:  # 对应的数字
        out = open("./newtrain.txt", 'w')
        con = i + "\\" + ii + "\n"
        lines.append(con)
        random.shuffle(lines)
        random.shuffle(lines)
        random.shuffle(lines)
        random.shuffle(lines)
        random.shuffle(lines)

for line in lines:
    out.write(line)
    number = number + 1
print("ok, the total is : " + str(number))
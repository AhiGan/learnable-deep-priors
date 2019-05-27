import argparse
import h5py
import numpy as np
import os
from common import create_dataset
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--folder_downloads')
parser.add_argument('--filename', default='shapes_20x20.h5')
args = parser.parse_args()

f = h5py.File(os.path.join(args.folder_downloads, args.filename), 'r')
images_prev = f['features'][()]
labels_mse_prev = f['mask'][()]
image_size = 20
# 以0、1的方式表示一张包含三个component的图像
images = images_prev.reshape(images_prev.shape[0], 1, image_size, image_size)[:-10000]
# 以0、0.33334、1表示不在该组、有可能在该组（被三组平分？）、在该组
labels_mse = labels_mse_prev.reshape(labels_mse_prev.shape[0], -1, 1, image_size, image_size)[:-10000]
# print("labels_mse shape before：", labels_mse.shape)
#
# with open('labels_mse2.csv','w') as f:
#     f_csv = csv.writer(f)
#     # f_csv.writerow(headers)
#     f_csv.writerows(labels_mse[0,2,0,:,:])
# print(labels_mse[0,1,0,:,0:10])

# print("images shape: ",images.shape)
# with open('image.csv','w') as f:
#     f_csv = csv.writer(f)
#     # f_csv.writerow(headers)
#     f_csv.writerows(images[0,0,:,:])
# print(images[0,0,:,:])
labels_mse *= images[:, None]   # 这个乘法是什么意思
# print(labels_mse[0,0,0,:,0:5])

# with open('labels_mse_after2.csv','w') as f:
#     f_csv = csv.writer(f)
#     # f_csv.writerow(headers)
#     f_csv.writerows(labels_mse[0,2,0,:,:])


labels_ami = np.ndarray(images.shape, dtype=images.dtype)
mask = np.zeros(images.shape, dtype=np.int)

# 第i个group
for i in range(labels_mse.shape[1]):
    pos_sel = labels_mse[:, i] != 0   # pos_sel的shape是（labels_mse.shape[0]，1，image_size, image_size）内容是True/False
    # if i == 0:
    #     print(pos_sel.shape)
    #     print("pos_sel: ", pos_sel[0,0,:,:])
    #     print(labels_ami.shape)
    #     print("labels_ami_before: ", labels_ami[0,0,:,:])

    labels_ami[pos_sel] = i + 1

    # if i ==0:
    #     print(labels_ami.shape)
    #     print("labels_ami_after: ", labels_ami[0,0,:,:])
    #     print("mask_before: ", mask[0,0,:,:])

    mask[pos_sel] += 1
# print("labels_ami1: ", labels_ami[0, 0, :, :])
# print("mask_after_all: ", mask[0, 0, :, :])


# 去除单维度
labels_ami = labels_ami.squeeze(1)
# print("labels_ami2: ", labels_ami[0, :, :])

# print("labels_ami shape: ", labels_ami.shape)
# print("labels_ami: \n", labels_ami[0, :, :])
# print("labels_mse shape", labels_mse.shape)
# print("labels_mse_1 \n", labels_mse[0,0,:,:])
# print("labels_mse_2 \n", labels_mse[0,1,:,:])
# print("labels_mse_3 \n", labels_mse[0,2,:,:])

'''
images:  (70000, 1, 20, 20)
labels_ami:  (70000, 20, 20)
labels_mse:  (70000, 3, 1, 20, 20)
'''
sep1, sep2 = 50000, 60000
images = {'train': images[:sep1], 'valid': images[sep1:sep2], 'test': images[sep2:]}
labels_ami = {'train': labels_ami[:sep1], 'valid': labels_ami[sep1:sep2], 'test': labels_ami[sep2:]}
labels_mse = {'train': labels_mse[:sep1], 'valid': labels_mse[sep1:sep2], 'test': labels_mse[sep2:]}
create_dataset('shapes_20x20', images, labels_ami, labels_mse)

f.close()
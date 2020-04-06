import csv
import os
from imutils import paths
import random

img_folder = './dataset/UTK/'
# img_name_ck_list = list(paths.list_images(img_folder_ck))
img_name_list = os.listdir(img_folder)
random.shuffle(img_name_list)
num_img = len(img_name_list)
num_train = int(num_img*70/100)

with open('data_train.csv', 'a') as csvfile:

    writer = csv.writer(csvfile)
    for i in range(0, num_train):

        cls_gt = int(img_name_list[i].split('_')[0])
        img_path = img_folder + img_name_list[i]
        writer.writerow([img_path, cls_gt])

with open('data_test.csv', 'a') as csvfile:

    writer = csv.writer(csvfile)
    for i in range(num_train, num_img):

        cls_gt = int(img_name_list[i].split('_')[0])
        img_path = img_folder + img_name_list[i]
        writer.writerow([img_path, cls_gt])
        

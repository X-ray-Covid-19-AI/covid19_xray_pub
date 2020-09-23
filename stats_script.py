import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import shutil

file_dir =  "ieee8023_data/"

pos_dir = os.path.join(file_dir, "covid_images")
neg_dir = os.path.join(file_dir, "noncovid_images")
pos_ims = os.listdir(pos_dir)
neg_ims = os.listdir(neg_dir)

widths = []
heights = []
for im_name in pos_ims:
    full_path = os.path.join(pos_dir,im_name)
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    img_height = img.shape[0]
    img_width  =  img.shape[1]
    widths.append(img_width)
    heights.append(img_height)

plt.hist(widths, bins = 55)
plt.hist(heights, bins = 55)
plt.show()

nwidths = []
nheights = []
for im_name in neg_ims:
    full_path = os.path.join(neg_dir,im_name)
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    img_height = img.shape[0]
    img_width  =  img.shape[1]
    nwidths.append(img_width)
    nheights.append(img_height)

plt.hist(nwidths, bins = 55)
plt.hist(nheights, bins = 55)
plt.show()

np.random.seed(1414)

num_neg = len(neg_ims)
num_pos = len(pos_ims)
percent_test = 0.15

neg_ims = np.random.permutation(neg_ims)
pos_ims = np.random.permutation(pos_ims)

train_neg = neg_ims[:int(num_neg*percent_test)]
test_neg = neg_ims[int(num_neg*percent_test)+1:]

train_pos = pos_ims[:int(num_pos*percent_test)]
test_pos = pos_ims[int(num_pos*percent_test)+1:]

train_pos_out_dir = "ieee8023_data/ieee8023_data_arranged/train/covid"
test_pos_out_dir = "ieee8023_data/ieee8023_data_arranged/test/covid"
test_neg_out_dir = "ieee8023_data/ieee8023_data_arranged/test/non_covid"
train_neg_out_dir = "ieee8023_data/ieee8023_data_arranged/train/non_covid"

os.mkdir("ieee8023_data/ieee8023_data_arranged")
os.mkdir("ieee8023_data/ieee8023_data_arranged/train")
os.mkdir("ieee8023_data/ieee8023_data_arranged/test")
os.mkdir(test_pos_out_dir)
os.mkdir("ieee8023_data/ieee8023_data_arranged/test/non_covid")
os.mkdir(train_pos_out_dir)
os.mkdir("ieee8023_data/ieee8023_data_arranged/train/non_covid")

for im in train_pos:
    filePath = os.path.sep.join([pos_dir, im])
    shutil.copy2(filePath, train_pos_out_dir)

for im in test_pos:
    filePath = os.path.sep.join([pos_dir, im])
    shutil.copy2(filePath, test_pos_out_dir)

for im in test_neg:
    filePath = os.path.sep.join([neg_dir, im])
    shutil.copy2(filePath, test_neg_out_dir)

for im in train_neg:
    filePath = os.path.sep.join([neg_dir, im])
    shutil.copy2(filePath, train_neg_out_dir)
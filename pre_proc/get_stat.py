import numpy as np
import pandas as pd
import cv2
import os
from matplotlib import pyplot as plt
def get_stat(in_path, out_path, in_data_fname, out_data_fname,extensions):
    '''

    :param in_path: releative path to input metadata and images
    :param out_path: releative path to output metadata
    :param in_data_fname: name of input metadata file
    :param out_data_fname: name of output metadata file to be generate
    :param extensions:  image extensions which will be scanned vs. metadata file names
    :return:   an output metadata file would be generated upon valid input.

    Description:
              receives input metadata file and path to images, genrates a corresponding output metadata file with statistics
              in turn, the output metadata file can be used for e.g. hypothesis testing of various kind of biases
              (e.g. using Pandera)
              in case an image file is not found, a NaN is given for the corresponding row statistic values
     Developer:
           Y.Shachar 6/2020.
    '''
    in_abs_path = os.path.abspath(in_path)
    fname = os.path.join(in_abs_path, in_data_fname)
    df = pd.read_csv(fname)
    # generate 2 lists: 1. of filenames (without extensions)  from metadata
    # 2. of corresponding image filenames from input directory
    list1 = df['FILE_NAME'].tolist()
    list2 = [f for f in os.listdir(in_abs_path) if
             os.path.isfile(os.path.join(in_abs_path, f)) and f[0:f.find('.')] in list1 and f[f.rfind('.'):] in extensions]
    # empty dictionary
    mean_dict = {}
    std_dict  = {}
    print ('* * List of Image files: * *')
    for f1 in list1:
        for f2 in list2:
            if f1 in f2: # if there is a matching image file. read image and get statistics
                full_path = os.path.join(in_abs_path,f2)
                print(full_path)
                img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                img_height = img.shape[0]
                img_width  =  img.shape[1]
                if img_height > 0 and img_width > 0: # read image successfully
                    mean, stddev = cv2.meanStdDev(img/255.0) # normalize values to 0 -1.
                    mean_dict[f1] =  round(mean.item(),3) # round 3 decimals right to "."
                    std_dict[f1]  =  round(stddev.item(),3)
    # generate data frames for mean and std
    df_mean  = pd.DataFrame.from_dict(mean_dict, orient='index',columns=[ 'Mean'])
    df_std   = pd.DataFrame.from_dict(std_dict , orient='index',columns=[ 'Std'])
    #print(df_mean)
    #print(df_std)
    df_mean  = df_mean.reset_index()
    df_std   = df_std.reset_index()
    df_mean  = df_mean.rename({'index': 'FILE_NAME'}, axis=1)
    df_std   = df_std.rename({'index': 'FILE_NAME'}, axis=1)

    # merge with input data frame
    df_res = df.merge(df_mean, how='outer', on='FILE_NAME')
    df_res = df_res.merge(df_std, how='outer', on='FILE_NAME')
    print('* * Output Data Frame: * *')
    print (df_res)

    # generate output metadata file
    out_abs_path = os.path.abspath(out_path)
    if (os.path.isdir(out_abs_path) == False):
        os.mkdir(out_abs_path)
    out_fname_path = os.path.join(out_abs_path, out_data_fname)
    df_res.to_csv(out_fname_path, index=False)
### unit test code ###
extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
get_stat(os.curdir+'\\In', os.curdir+'\\Out', 'ahuva1.csv', 'stat.csv',extensions)

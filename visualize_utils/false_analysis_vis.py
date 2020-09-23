import pandas as pd
import math
import os.path as osp
from dnn.typing_definitions import *


def get_fig(root_dir: DirPath, df: pd.DataFrame, col_num: int) -> Image:
    img_num = df.shape[0]
    row_num = round(math.ceil(img_num / col_num))
    from PIL import Image
    total_img = Image.new('L', (1024*col_num, 1024*row_num))
    for row_idx in range(row_num):
        row_imgs = Image.new('L', (1024*col_num, 1024))
        for col_idx in range(col_num):
            img_idx = row_idx * col_num + col_idx
            if img_idx < img_num:
                img_path = osp.join(root_dir, df.iloc[img_idx]['image_name'])
                img = Image.open(img_path)
                row_imgs.paste(img, (1024*col_idx, 0))
            else:
                continue
        total_img.paste(row_imgs, (0, 1024*row_idx))
    return total_img


def save_net_mistakes(results: pd.DataFrame, root_path: DirPath, col_num: int, output_dir: DirPath):
    """
    Plot images which the net did a mistake on
    :param results: the csv file which given from the net
    :param root_path: image root path (where all the images are)
    :param col_num: number of images to show in each column
    :param output_dir: dir to save images
    """
    mistake_df = results[results['gt'] != results['net_prediction']]
    false_positive_df = mistake_df[mistake_df['gt'] == 1]
    false_negative_df = mistake_df[mistake_df['gt'] == 0]
    fp_fig = get_fig(root_dir=osp.join(root_path, 'normal'), df=false_positive_df, col_num=col_num)
    fn_fig = get_fig(root_dir=osp.join(root_path, 'corona'), df=false_negative_df, col_num=col_num)
    fp_fig.save(osp.join(output_dir, 'false_positive.jpg'))
    fn_fig.save(osp.join(output_dir, 'false_nositive.jpg'))

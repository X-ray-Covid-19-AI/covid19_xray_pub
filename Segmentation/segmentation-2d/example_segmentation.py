from Demo import demo
from pathlib import Path
import glob
import pandas as pd
import cv2
DATA_DIR = "/home/ayelet/workspace/segmentation_input/"
OUT_DIR = "/home/ayelet/workspace/segmentation_output1/"
filenames = glob.glob('{}*.jpg'.format(DATA_DIR))
filenames = [f.replace(DATA_DIR,"") for f in filenames]
df = pd.DataFrame(filenames)
data_file = "{}file_stats.csv".format(DATA_DIR)

df.to_csv(data_file,header=["img"],index=False)
Path(OUT_DIR).mkdir(parents=True,exist_ok=True)

gen=demo.mask_generator(data_file,OUT_DIR,DATA_DIR,(1024,1024))
for i in gen:
    cv2.imshow('image', i)
    cv2.waitKey(0)



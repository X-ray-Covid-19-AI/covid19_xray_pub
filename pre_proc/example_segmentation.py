from Demo.demo import get_mask
import cv2

from pathlib import Path
import pandas as pd
import cv2
import os
import importlib
DATA_DIR =os.curdir+'\\InSegmentation'
OUT_DIR = os.curdir+'\\OutSegmentation'
#filenames = glob.glob('{}*.jpg'.format(DATA_DIR))
filenames = [name for name in os.listdir(DATA_DIR) if name.endswith(".jpg")]
filenames = [f.replace(DATA_DIR,"") for f in filenames]
df = pd.DataFrame(filenames)
#data_file = "{}file_stats.csv".format(DATA_DIR)
data_file = "{}idx.csv".format(DATA_DIR)

df.to_csv(data_file,header=["img"],index=False)
Path(OUT_DIR).mkdir(parents=True,exist_ok=True)
mask = get_mask(img)
gen=demo.mask_generator(data_file,OUT_DIR,DATA_DIR,(1024,1024))
for i in gen:
    cv2.imshow('image', i)
    cv2.waitKey(0)



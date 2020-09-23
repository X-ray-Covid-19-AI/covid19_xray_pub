import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import shutil
import pandas as pd
from keras import backend as be
from keras.models import load_model
from pathlib import Path
from skimage import morphology, color, exposure
from Demo import mygen

IMAGE_SHAPE = (256, 256)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(process)d - %(asctime)s - %(pathname)s - %(levelname)s - %(message)s')




def masked(img,im_tiny, mask, alpha=1,padding_rows=0,padding_cols=0):
    """Returns image with GT lung field outlined with red, predicted lung field
    filled with blue."""
    rows, cols = im_tiny.shape
    color_mask = np.zeros((rows, cols, 3))
    color_mask[mask == 1] = [0, 0, 1]
    img_color = np.dstack((im_tiny, im_tiny, im_tiny))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    z=np.argwhere(mask==1)
    rows=z[:,0]
    cols=z[:,1]
    if cols.size == 0 :
      cols = np.array([0,mask.shape[1]-1])
    if rows.size == 0 :
      rows = np.array([0,mask.shape[0]-1])
    c1,r1,c2,r2=min(cols),min(rows),max(cols),max(rows)
    img_masked = cv2.rectangle(img_masked,(c1,r1),(c2,r2),(0,255,0),3)

    rec_img = np.zeros((img.shape[0],img.shape[1]), np.uint8)
    row_factor=img.shape[0]/im_tiny.shape[0]
    col_factor=img.shape[1]/im_tiny.shape[1]
    x1=max(int(c1*col_factor)-padding_cols,0)
    x2=min(int(c2*col_factor)+padding_cols,rec_img.shape[1])
    y1=max(int(r1*row_factor)-padding_rows,0)
    y2=min(int(r2*row_factor)+padding_rows,rec_img.shape[0])
    logger.debug("x1:{},y1:{} x2:{},y2:{}".format(x1,y1,x2,y2))
    cv2.rectangle(rec_img,(x1,y1),
    (x2,y2),(255,255,255),thickness=-1)

    cut_image = cv2.bitwise_and(img, img, mask=rec_img)


    return img_masked,cut_image

def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img

def concat_images(imga, imgb,gap,bottom):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])+bottom
    total_width = wa+wb+gap
    new_img = np.ones(shape=(max_height, total_width,3))*255
    new_img[:ha,:wa]=imga
    new_img[:hb,wa+gap:wa+wb+gap]=imgb
    return new_img,wb

def concat_n_images(image_path_list,filename,names):
    """
    Combines N color images from a list of image paths.
    """
    output = None
    widths = []
    gap=20
    bottom=100
    for i, img_path in enumerate(image_path_list):
        img = plt.imread(img_path)
        #print("immg path: {} ,img_shape: {}".format(img_path,img.shape))
        if len(img.shape)<3:
          img = np.dstack((img,img,img))
          
        img=img*255
        #print(i)
        if i==0:
            output = img
            widths.append(img.shape[1])
        else:
            output,wb = concat_images(output, img,gap,bottom)
            widths.append(wb)
    y = int(output.shape[0]-bottom/2)
    start = 0
    for width,name in zip(widths,names):

      cv2.putText(output,name,(int(start+width/2),y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),1)
      start=start+width+gap
    cv2.imwrite(filename,output)

def prep(csv_path,outfolder,path,output_size=(1024,1024)):

    log_name = outfolder + "/log.txt"
    df = pd.read_csv(csv_path)
    ch = logging.FileHandler(log_name)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


    # Load test data
    composite_generator = mygen.myGenerator(csv_path,path,log_name,IMAGE_SHAPE,output_size)


    # Load model
    model_name = 'trained_model.hdf5'
    UNet = load_model(model_name)


    return composite_generator,df,UNet

def mask_generator(csv_path,outfolder,path,output_size):
    #output size for datasets with different image sizes
    composite_generator, df, UNet = prep(csv_path, outfolder, path,output_size)
    for original, xx in composite_generator:
        original = original.reshape(original.shape[1:3])

        pred = UNet.predict(xx)[..., 0].reshape(IMAGE_SHAPE)
        pred = cv2.resize(pred, original.shape)
        pr = pred > 0
        pr = remove_small_regions(pr, 0.02 * np.prod(pr.shape))
        pred[pr == 0] = 0

        yield pred


def write_comparative_to_file(csv_path,outfolder,path,padding_rows=50,padding_cols=50):
    composite_generator,df,UNet=prep(csv_path,outfolder,path)
    logger.debug('padding_rows:{},padding_cols:{}'.format(padding_rows,padding_cols))

    i = 0

    for original,xx in composite_generator:
        original = original.reshape(original.shape[1:3])

        pred = UNet.predict(xx)[..., 0].reshape(IMAGE_SHAPE)
        pred = cv2.resize(pred,original.shape)
        pr = pred > 0.5

        pr = remove_small_regions(pr, 0.02 * np.prod(pr.shape))


        filled,cut_image=masked(original,original ,pr, 1,
                padding_rows=padding_rows,padding_cols=padding_cols)
        print(filled.shape,cut_image.shape)
        target_file = outfolder + '/' +df.iloc[i][0]
        full_images_folder = target_file[:target_file.rfind('.')]
        Path(full_images_folder).mkdir(parents=True, exist_ok=True)


        original_name = "{}/original.png".format(full_images_folder)
        cropped_name = "{}/cropped.png".format(full_images_folder)
        filled_name = "{}/filled.png".format(full_images_folder)
        compare_name = "{}/compare.png".format(full_images_folder)

        cv2.imwrite(original_name,255*original)
        cv2.imwrite(cropped_name,255*cut_image)
        cv2.imwrite(filled_name,255*filled)

        concat_n_images([original_name,cropped_name,filled_name],
        target_file,["original","cropped","filled"])

        shutil. rmtree(full_images_folder)
        i+=1
        be.clear_session()

        if i==df.shape[0]:
          break


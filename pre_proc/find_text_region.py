'''
StandAlone unit test code for the process of cropping non relevant image side strips, which may cause bias,
 out of each the image in the input path :
1) finding and removing text area from an auxiliary copy of the image, and replace with dark values.
2) following step (1), find dark, non relevant top,bottom,left,right strips in the auxiliary image.
3) crop the original image such that these strips will be  discarded.
This code requires that tesseract will be installed and the PATH env. variable, should contain the folder where
the executable tesseract file resides. In addition, the python wrapper to tesseract, namely pytesseract should be
installed in the python environment.
'''
import cv2
import numpy as np
from PIL import Image
import pydicom as dicom
import os
import pytesseract
'''
Find top border line of dark area in the image.
Make an Histogram on a given horizontal strip, and continue, as long
as the portion of the lowest 1/5 of possible intensity values, is at least hist_threshold
img - image to find dark top strip
hist_threshold - portion of the strip to be dark, in order for the strip to be considered for cropping.
strip_size - stride for the strip test.
'''

def find_black_top(img,hist_threshold = 0.8,strip_size = 5):
    height = img.shape[0]
    width = img.shape[1]
    if len(img.shape) > 2:
        depth = img.shape[2]
    else:
        depth = 1
    if strip_size < 1:
        return 0
    if height < 10*strip_size or width < 10:
        return 0
    top_line = 0
    for i in range (0,height-strip_size,strip_size):
        if (depth == 1):
            counts, bin_edges = np.histogram(img[i:i+strip_size,:], range=(0, 255))
        else:
            counts, bin_edges = np.histogram(img[i:i + strip_size, :, :], range=(0, 255))
        if (counts[0] + counts[1] < width*depth*strip_size*hist_threshold):
            break
        #print('count for line:', top_line, 'is ', counts[0] + counts[1],'with width:',width)
        top_line += strip_size
    return min(top_line,height*4//10)
'''
same as above, for the bottom part of the image
'''
def find_black_bottom(img,hist_threshold = 0.8,strip_size = 5):
    height = img.shape[0]
    width = img.shape[1]
    if len(img.shape) > 2:
        depth = img.shape[2]
    else:
        depth = 1
    if strip_size < 1:
        return height - 1
    if height < 10*strip_size or width < 10:
        return height - 1
    bottom_line = height-1
    for i in range (height-strip_size-1,0,-strip_size):
        if (depth == 1):
            counts, bin_edges = np.histogram(img[i:i+strip_size,:], range=(0, 255))
        else:
            counts, bin_edges = np.histogram(img[i:i + strip_size, :, :], range=(0, 255))
        if (counts[0] + counts[1] < width*depth*strip_size*hist_threshold):
            break
        #print('count for line:', bottom_line, 'is ', counts[0] + counts[1],'with width:',width)
        bottom_line -= strip_size
    return max(bottom_line,height*6//10)
'''
same as above, for the left part of the image
'''
def find_black_left(img,hist_threshold = 0.9,strip_size = 5):
    height = img.shape[0]
    width = img.shape[1]
    if len(img.shape) > 2:
        depth = img.shape[2]
    else:
        depth = 1
    if strip_size < 1:
        return 0
    if width < 10*strip_size or height < 10:
        return 0
    left_column = 0
    for i in range (0,height-strip_size,strip_size):
        if (depth == 1):
            counts, bin_edges = np.histogram(img[:,i:i+strip_size], range=(0, 255))
        else:
            counts, bin_edges = np.histogram(img[:,i:i + strip_size,:], range=(0, 255))
        if (counts[0] + counts[1] < height*depth*strip_size*hist_threshold):
            break
        #print('count for line:', top_line, 'is ', counts[0] + counts[1],'with width:',width)
        left_column += strip_size
    return min(left_column,width//3)
'''
same as above, for the right part of the image
'''
def find_black_right(img,hist_threshold = 0.9,strip_size = 5):
    height = img.shape[0]
    width = img.shape[1]
    if len(img.shape) > 2:
        depth = img.shape[2]
    else:
        depth = 1
    if strip_size < 1:
        return width - 1
    if width < 10*strip_size or height < 10:
        return width - 1
    right_column = width-1
    for i in range (width - strip_size-1,0,-strip_size):
        if (depth == 1):
            counts, bin_edges = np.histogram(img[:,i:i+strip_size], range=(0, 255))
        else:
            counts, bin_edges = np.histogram(img[i:i + strip_size,:,:], range=(0, 255))
        #print('count for column:', right_column, 'is ', counts[0] + counts[1], 'with height:', height)
        if (counts[0] + counts[1] < height*depth*strip_size*hist_threshold):
            break

        right_column -= strip_size
    return max(right_column,width*2//3)


# find text areas in the image and mask them. Then remove top and bottom dark parts.
'''
get an input image, and return an image which is a cropped version of 
the input image, without non relevant side strips.
'''
def find_regions(img):
    #convert to gray and pass through threshold
    if (len(img.shape) > 2):
        img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img2gray = img.copy()
    ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY)
    image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
    ret, new_img = cv2.threshold(image_final, 200, 255, cv2.THRESH_BINARY)  # for black text , cv.THRESH_BINARY_INV
    # dilate the image to get contours:
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,
                                                         3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    dilated = cv2.dilate(new_img, kernel, iterations=9)  # dilate , more the iteration more the dilation

    # for cv2.x.x

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # findContours returns 3 variables for getting contours

    # for cv3.x.x comment above line and uncomment line below

    #image, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #prepare black mask
    mask = np.zeros(img2gray.shape, np.uint8)
    index = 1
    # go over each contour, masking its bounding rect, if it contains text.
    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # Don't process  small/very large false positives that aren't text
        if w < 20 and h < 15:
            continue
        if h > img2gray.shape[0]/5 and w > img2gray.shape[1]/3:
            continue
        # for debug: draw rectangle around contour on original image
        #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)


        # crop threshold image and send to OCR, if it returns text, mask with black)
        cropped = image_final[y :y +  h , x : x + w]
        roi = Image.fromarray(cropped)
        text = pytesseract.image_to_string(roi)
        #cv2.imshow('ImageWindow', img2gray)
        #cv2.waitKey(0)
        if (len(text)>0): # text area, blacken this roi
            img2gray[y :y +  h , x : x + w] = mask[y :y +  h , x : x + w]
        #s = file_name.strip('.tif') + '_crop_' + str(index) + '.jpg'

        index = index + 1 # just for debug

    htop = find_black_top(img2gray)
    hbottom = find_black_bottom(img2gray)
    img2gray = img2gray[htop:hbottom,:]
    wright  = find_black_right(img2gray,hist_threshold = 0.8,strip_size = 5)
    wleft   = find_black_left(img2gray)

    img = img[htop:hbottom,wleft:wright]
    print('remove top area till line: ', htop)
    print('remove bottom area from line: ', hbottom)
    print('remove left  area till line: ', wleft)
    print('remove right area from line: ', wright)
    return (img)
'''
provide batch region removal, for all the relevant 
files in the input path with the proper specified extensions
and write cropped files into the out_path.
when petct_filter is set, files with names containing 'pet' and 'ct'
will be ignored. 
'''
def remove_regions_batch(in_path, out_path, extensions, min_size, petct_filter):
    in_abs_path = os.path.abspath(in_path)
    files = [f for f in os.listdir(in_abs_path) if \
             os.path.isfile(os.path.join(in_abs_path, f)) and f[f.rfind('.'):] in extensions]
    for f in files:
        fname = os.path.join(in_abs_path, f)
        if (fname.find('_crop') != -1):
            continue  # do not process images which are already processed
        if (petct_filter):  # filter out pet ct images
            fl = f.lower()
            if (fl.find('pet') != -1) and fl.find('ct') != -1:
                continue
        ext = f[f.rfind('.'):]
        out_ext = ext
        # if input image is dcm, we use dicom to read, and it will be saved as tiff.
        if '.dcm' == ext:
            ds = dicom.dcmread(fname)
            in_img = ds.pixel_array
            out_ext = '.tiff'
        else:
            in_img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        if len(ext) > 0:
            fs = f[:-len(ext)]
        else:
            fs = f
        print (' *** finding text regions for file:',fs,'  ***')
        out_img = find_regions(in_img)
        out_abs_path = os.path.abspath(out_path)
        if (os.path.isdir(out_abs_path) == False):
            os.mkdir(out_abs_path)
        out_fname_path = os.path.join(out_abs_path, fs)
        out_fname = out_fname_path + '_crop' + out_ext
        if (out_img.shape[0] < min_size or out_img.shape[1] < min_size):
            print('#### not writing file name: ', out_fname, ' at least one img dim is too small ###')
        else:
            print ('#### writing file name: ', out_fname, ' ###' )
            cv2.imwrite(out_fname, out_img)
''' Unit Test'''
## unit test code ###
if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    petct_filter = True
    extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff','.dcm']
    min_size = 400
    # call with input path, output path, extensions list ( use sub list of the above), and petct filter
    # this script will crop non relevant regions in input files reside in the input path with the proper extension.
    remove_regions_batch(os.curdir+'\\InCrop1', os.curdir+'\\OutCrop', extensions, min_size, petct_filter)

import numpy as np
import cv2
import pydicom as dicom
import os
import pytesseract
from pre_proc.Demo.demo import get_mask
from matplotlib import pyplot as plt
import pre_proc.augment_data as aug
from argparse import ArgumentParser
from pre_proc.find_text_region import find_regions


class Ctr:

    def __init__(self):
        self.TotalFilesCount = 0
        self.NonQualifiedFilesCount = 0

    def inc_total(self):
        self.TotalFilesCount += 1

    def inc_unqualified(self):
        self.NonQualifiedFilesCount += 1

    def print(self):
        print(' Total File Count:', self.TotalFilesCount)
        print(' Non Qualified File Count:', self.NonQualifiedFilesCount)


ctrs = Ctr()


# --------------------------------------------------------------
def qualify_image(in_img, fsize, min_pixels):
    '''
    qualifies image if minimal requirements are met:
            a. at least min_pixels for rows and columns.
            b. file size reflects loss-less or almost loss-less compression.
            c. an absolute minimal file size, based on min_pixels, since for small images
               the loss-less compression ratio should be close to 1.
    :param in_img: input image to be tested.
    :param fsize:  indicated file size of the image file
    :param min_pixels: min # pixels in each direction
    :return:  bool - True if qualified,  False otherwise
    '''
    CompressionRatioSmall = 1.1
    CompressionRatioRegular = 2.0
    quality = False
    if (in_img.shape[0] >= min_pixels) and (in_img.shape[1] >= min_pixels):
        min_fsize1 = round(float(min_pixels * min_pixels) / CompressionRatioSmall)
        min_fsize2 = round(float(in_img.shape[0] * in_img.shape[1]) / CompressionRatioRegular)
        if (fsize >= min_fsize1) and (fsize >= min_fsize2):
            quality = True
    return quality


# --------------------------------------------------------------
def normalizeScaleSegment(in_img, height, width, scale, segmentation, clahe):
    if (clahe):
        # w'd rather not use global normalization, as it will reduce the color depth of
        # "dense" histogram area. Instead, we use the CLAHE method
        # see https://en.wikipedia.org/wiki/Adaptive_histogram_equalization
        # create a CLAHE object (Arguments are optional), and apply it
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe_obj.apply(in_img)
    else:
        clahe_img = in_img
    if (scale):
        img_height = clahe_img.shape[0]
        img_width = clahe_img.shape[1]
        if (img_height != height or img_width != width) and (height > 0 and width > 0):
            height_ratio = height / img_height
            width_ratio = width / img_width
            scale_ratio = min(height_ratio, width_ratio)
            # dsize
            dsize = (int(img_width * scale_ratio), int(img_height * scale_ratio))

            # resize image
            intrim_img = cv2.resize(clahe_img, dsize, interpolation=cv2.INTER_AREA)
            intrim_img_height = intrim_img.shape[0]
            intrim_img_width = intrim_img.shape[1]
            top_padding = int((height - intrim_img_height) / 2)
            bottom_padding = int((height - intrim_img_height + 1) / 2)
            left_padding = int((width - intrim_img_width) / 2)
            right_padding = int((width - intrim_img_width + 1) / 2)
            scaled_img = cv2.copyMakeBorder(intrim_img, top_padding, bottom_padding, left_padding, right_padding,
                                            cv2.BORDER_CONSTANT, 0)
        else:
            scaled_img = clahe_img
    else:
        scaled_img = clahe_img
        # do segmentation, and insert it as a second channel
    if (segmentation):
        mask = get_mask(scaled_img)
        assert (mask.dtype == np.float32)
        assert (mask.shape[0] == scaled_img.shape[0])
        assert (mask.shape[1] == scaled_img.shape[1])
        out_img = np.zeros((scaled_img.shape[0], \
                            scaled_img.shape[1], 3), \
                           dtype=np.uint8)
        mask = mask / mask.max()  # normalizes data in range 0 - 1
        mask = 255 * mask
        out_img[:, :, 0] = scaled_img
        out_img[:, :, 1] = mask.astype(np.uint8)
    else:
        out_img = scaled_img

    return out_img


# ---------------------------------------------------------
def preproc_single_folder(in_path, out_path, args, data_frame=''):
    """
    Name: normalize
    :param in_path:      path to the input image to be read
    :param out_path:     path to the output image to be written
    :param args - namespace containing:
           extensions:   list of image file extensions to be read (e.g. 'tif', 'tiff' - no dot needed!)
           height:       height for output image
           width:        width for  output image
           show:         bool flag, resulted normalization and hist plot will be shown for True
           write:        bool flag, normalized image file will be written for True
           segmentation: bool flag, generate  segmentation mask for input image if True
           augmentation: bool flag, generate augmentation images for input image if True
           clahe:        bool flag, perform CLAHE if True
    :return:           none.
    Description:
        normalization of Image files with the specified extensions in the current path.
        input image will be converted to 1 channel 8 bit Uchar image with the resulted sized height*width.
        if clahe is set, Adaptive histogram normalization algorithm will be performed, based on CLAHE.
        Contrast Limited Adpative Hist. Equalization )
        see https://en.wikipedia.org/wiki/Adaptive_histogram_equalization
        scaling will be perform to the specified width and height, while maintaining the aspect
        ratio, adding symmetric black padding, if required.
        if segmentation flag is specified, a segmentation mask will be generated, and added
        as a second channel. Output image will have 3 channels in such a case.
     Developer:
           Y.Shachar 5/2020.
    """
    print("preproc_single_folder: in_path: " + in_path + ", out_path: " + out_path)

    in_abs_path = os.path.abspath(in_path)
    # aug1 = aug.get_augs()
    files = [f for f in os.listdir(in_abs_path) if
             os.path.isfile(os.path.join(in_abs_path, f)) and f.rsplit('.')[-1] in args.extensions]
    for f in files:
        ctrs.inc_total()
        fname = os.path.join(in_abs_path, f)
        # 30/08/20 - norm filter removed, since in and out paths, must be disjoint
        # if (fname.find('_norm') != -1):
        #    continue # do not process images which are already processed
        fl = f.lower()
        if (args.petct_filter):  # filter out pet ct images

            if (fl.find('pet') != -1) or fl.find('ct') != -1:
                print(f, ": file name hints for a Pet CT Image, skipping...")
                continue

        if (args.lat_filter):  # filter lateral  images

            if (fl.find('lat') != -1):
                print(f, ": file name hints for a Lateral Image, skipping...")
                continue

        ext = f[f.rfind('.'):]
        out_ext = ext
        if '.dcm' == ext:
            ds = dicom.dcmread(fname)
            in_img = ds.pixel_array
            if ds.BitsStored > 8:
                in_img = (in_img >> (ds.BitsStored - 8)).astype('uint8')
            if ds.PhotometricInterpretation == 'MONOCHROME1':
                in_img = cv2.bitwise_not(in_img)
            out_ext = '.tiff'
            if (args.lat_filter):  # filter lateral  images
                try:
                    acdpl = ds.AcquisitionDeviceProcessingDescription.lower()
                    if (acdpl.find('lat') != -1):
                        print(f, ": file contains a lateral Image, skipping...")
                        continue
                except:
                    print(f, ": warning- Acuisition Device Processing Desc not found")
        else:
            in_img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        try:
            width1 = in_img.shape[1]
        except:
            print('warning: non image file: ', fname, '...skipping')
            continue
        if args.invert:
            in_img = cv2.bitwise_not(in_img)
            print('inverting data for image file: ', fname)
        if args.crop_text:
            print('cropping text regions for image file: ', fname)
            in_imgc = find_regions(in_img)
        else:
            in_imgc = in_img
        fsize = os.stat(fname).st_size
        if args.quality_filter:
            is_qualified = qualify_image(in_imgc, fsize, args.min_pixels)
            if not (is_qualified):
                print('warning: image file: ', fname, ' not qualified...skipping')
                ctrs.inc_unqualified()
                continue
        if len(ext) > 0:
            fs = f[:-len(ext)] + '_norm'
        else:
            fs = f + '_norm'
        # fs = f.strip(ext)
        img_dict = {fs: in_imgc}
        # aug1 = aug.get_augs_done(image=[in_img])
        if (args.augmentation):
            aug_function = aug.get_augs_done
            # all_augs_done_df, new_images_dict = aug.augment_images(aug1, [in_img], [fs], ITERS=['A', 'B', 'C', 'D'])
            # _,new_images_dict = aug.augment_images(aug1, [in_img], [fs], ITERS=['A', 'B', 'C', 'D'])
            new_images_dict, all_augs_done_df = aug.augment_images(aug_function, [in_imgc], [fs],
                                                                   ITERS=['A', 'B', 'C', 'D'])
            img_dict.update(new_images_dict)
            # print('using augmentation list:', all_augs_done_df)

        for key in img_dict:
            img = img_dict[key]
            out_img = normalizeScaleSegment(img, args.height, args.width, True, args.segmentation, args.clahe)

            if (args.show):
                cv2.imshow('ImageWindow', out_img)
                cv2.waitKey()
            if (args.write):
                out_abs_path = os.path.abspath(out_path)
                if (os.path.isdir(out_abs_path) == False):
                    os.mkdir(out_abs_path)
                # out_fname_path = os.path.join(out_abs_path, os.path.splitext(f)[0])
                out_fname_path = os.path.join(out_abs_path, key)
                out_fname = out_fname_path + out_ext

                cv2.imwrite(out_fname, out_img)

    if (args.show):
        plt.close('all')


########################################################################################################################

def process_folder_recursive(base_folder_source, base_folder_target, args, subfolder_name=''):
    print(
                "process_folder_recursive: source: " + base_folder_source + ", target: " + base_folder_target + ", subfolder: " + subfolder_name)
    #####################################################
    # process all files in current dir
    curr_source = os.path.join(base_folder_source, subfolder_name)
    curr_target = os.path.join(base_folder_target, subfolder_name)

    if not os.path.exists(curr_target):
        os.makedirs(curr_target)
        preproc_single_folder(in_path=curr_source, out_path=curr_target, args=args)
    else:
        if args.overwrite:
            os.makedirs(curr_target, exist_ok=True)
            preproc_single_folder(in_path=curr_source, out_path=curr_target, args=args)
        else:
            print("Folder " + curr_target + " exists and overwrite not selected, ignoring...")

    #####################################################
    # process recursion
    curr_contents = os.listdir(curr_source)
    # print(curr_contents)

    for item in curr_contents:
        print("Checking " + os.path.join(base_folder_source, subfolder_name, item))
        if os.path.isdir(os.path.join(base_folder_source, subfolder_name, item)):
            new_subfolder = os.path.join(subfolder_name, item)
            print("Going into subfolder " + new_subfolder + "...")
            process_folder_recursive(base_folder_source, base_folder_target, args, new_subfolder)
    print("LENGTH  FO CHECKING IS: ", str(len(curr_contents)))


########################################################################################################################

def fill_parser(parser: ArgumentParser):
    '''
    Use this from external file
    :return:
    '''

    parser.add_argument('input_folder', help='Input folder to preprocess',
                        default=os.path.join(os.curdir, 'pre_proc', 'In'))
    parser.add_argument('output_folder', help='Output folder to save preprocessed data',
                        default=os.path.join(os.curdir, 'pre_proc', 'In'))

    parser.add_argument('-extensions', nargs='+', default=['png', 'jpg', 'jpeg', 'tif', 'tiff', 'dcm'],
                        help='image extensions to process: -extensions png jpeg jpg tif tiff')
    parser.add_argument('--segmentation', dest='segmentation', action='store_true')
    parser.add_argument('--no-segmentation', dest='segmentation', action='store_false')
    parser.set_defaults(segmentation=True)

    parser.add_argument('--augmentation', dest='augmentation', action='store_true')
    parser.add_argument('--no-augmentation', dest='augmentation', action='store_false')
    parser.set_defaults(augmentation=True)

    parser.add_argument('--clahe', dest='clahe', action='store_true')
    parser.add_argument('--no-clahe', dest='clahe', action='store_false')
    parser.set_defaults(clahe=True)

    parser.add_argument('--show', dest='show', action='store_true')
    parser.add_argument('--no-show', dest='show', action='store_false')
    parser.set_defaults(show=False)

    parser.add_argument('--write', dest='write', action='store_true')
    parser.add_argument('--no-write', dest='write', action='store_false')
    parser.set_defaults(write=True)

    parser.add_argument('--petct_filter', dest='petct_filter', action='store_true')
    parser.add_argument('--no-petct_filter', dest='petct_filter', action='store_false')
    parser.set_defaults(petct_filter=True)

    parser.add_argument('-height', type=int, default=1024)
    parser.add_argument('-width', type=int, default=1024)

    parser.add_argument('-min_pixels', type=int, default=600)
    parser.add_argument('--quality_filter', dest='quality_filter', action='store_true')
    parser.add_argument('--no-quality_filter', dest='quality_filter', action='store_false')
    parser.set_defaults(quality_filter=True)

    parser.add_argument('--crop_text', dest='crop_text', action='store_true')
    parser.add_argument('--no-crop_text', dest='crop_text', action='store_false')
    parser.set_defaults(crop_text=False)

    parser.add_argument('--lat_filter', dest='lat_filter', action='store_true')
    parser.add_argument('--no-lat_filter', dest='lat_filter', action='store_false')
    parser.set_defaults(lat_filter=False)

    parser.add_argument('--invert', dest='invert', action='store_true')
    parser.add_argument('--no-invert', dest='invert', action='store_false')
    parser.set_defaults(invert=False)

    parser.add_argument('--overwrite', dest='overwrite', action='store_true')

    return parser


def get_args():
    """
    Use this when calling from __main__
    :return:
    """
    parser = ArgumentParser()
    fill_parser(parser)
    return parser.parse_args()


def main(args):
    pytesseract.pytesseract.tesseract_cmd = '/apps/RH7U2/gnu/tesseract/1.0/bin/tesseract'
    in_abs_path = os.path.abspath(args.input_folder)
    out_abs_path = os.path.abspath(args.output_folder)
    if (out_abs_path.find(in_abs_path) != -1):
        print("Error: Outpath included in Inpath, Exiting...")
    else:
        process_folder_recursive(args.input_folder, args.output_folder, args=args)
        ctrs.
        print()


if __name__ == '__main__':
    args = get_args()
    main(args)
    ### unit test code ###
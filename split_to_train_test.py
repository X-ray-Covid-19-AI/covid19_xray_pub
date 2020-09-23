import random
from itertools import chain 
from shutil import move, copyfile
import glob
import os
import pickle
from argparse import ArgumentParser

###### ARGUMENTS #######
# PATH_PICKLE = 'splitted_train_test.pickle

"""
This function get file as input. It is mandatory that whithin the directory the positive images will be storred in files with namw "covid" and negative images storred in directory with the name "non_covid". The script serched recursively and read only directories with thes names.
"""
"""
TODO:
- Add path to pickle mandatory when using 'update'
- Option to update pickle with no 'val', and crate output with 'val'
"""
    
def save_pickle(dict_covid, dict_non_covid):
    """
    save dictionary as pickle file.
    The dictionary looks as follows: {covid: {train:images, test:images, val(optioanl):images}, non_covid: {train:images, test:images, val(optioanl):images}}
    """
    splitted_dict = {'dict_covid': dict_covid, 'dict_non_covid': dict_non_covid}
    out_fname = os.path.join(args.output_folder, 'splitted_train_test.pickle')
    with open(out_fname, 'wb') as f:
        pickle.dump(splitted_dict, f, pickle.HIGHEST_PROTOCOL)

        
def save_list(dict_covid, dict_non_covid):
    """
    save a txt file with information of the split: Random_state used, number of files in each directory, lists of files. 
    """
    out_fname = os.path.join(args.output_folder, 'train_test_splitted.txt')
    with open(out_fname , 'w') as filehandle:
        filehandle.write('Random state for splitting was: {f}\n\n'.format(f = str(RANDOM_STATE))) # random state
        
        for key in dict_covid.keys():
            filehandle.write('{key} covid files: {f} \n'.format(key = key, f = str(len(dict_covid[key])))) # number of files
            filehandle.write('{key} non_covid files: {f}\n'.format(key = key, f = str(len(dict_non_covid[key]))))
        for key, files in dict_covid.items():
            filehandle.write('\n{key} COVID FILES:\n\n'.format(key = key.upper())) # listst of covid files saved
            filehandle.writelines("%s\n" % file for file in files)
        for key, files in dict_non_covid.items():
            filehandle.write('\n{key} NON_COVID FILES:\n\n'.format(key = key.upper())) # listst of non_covid files saved
            filehandle.writelines("%s\n" % file for file in files)

            
def move_files(dictionary, label):
    """
    create symlink original directory to destinattion directory. 
    Before copying, destination directories are created for train, test and val, for covid and non_covid.  
    """
    out_abs_path = os.path.abspath(args.output_folder)
    if (os.path.isdir(out_abs_path) == False): #create destination directories
        os.mkdir(out_abs_path)
        os.mkdir(os.path.join(out_abs_path,'train'))
        os.mkdir(os.path.join(out_abs_path,'train', 'covid'))
        os.mkdir(os.path.join(out_abs_path,'train', 'non_covid'))
        os.mkdir(os.path.join(out_abs_path,'test'))
        os.mkdir(os.path.join(out_abs_path,'test', 'covid'))
        os.mkdir(os.path.join(out_abs_path,'test', 'non_covid'))
        if(args.validation):
            os.mkdir(os.path.join(out_abs_path,'val'))
            os.mkdir(os.path.join(out_abs_path,'val', 'covid'))
            os.mkdir(os.path.join(out_abs_path,'val', 'non_covid'))
    
    for key, list_value in dictionary.items(): # copy list of files
        for file in list_value:
            file_name = file.split(sep = "/")[-1]
            destination = args.output_folder + '/' + key + '/' + label + '/' + file_name
            #file = os.readlink(file)
            #os.symlink(file, destination)
            try:
                os.symlink(file, destination)
            except OSError as e:
                print("link exists. skipping...")   
                pass
            

def balance_data(dict1, dict2):
    """
    Balance positive and negative data by comparing the length of each dataset and choosing the first x
    files of the longer, so that both sets will be equal in length.
    """
    print("Balance data")
    for key1, key2 in zip(dict1, dict2):
        if(len(dict1[key1]) > len(dict2[key2])):  
            dict1[key1] = dict1[key1][:len(dict2[key2])]
        else: 
            dict2[key2] = dict2[key2][:len(dict1[key1])]

    return dict1, dict2


def concat_dictionaries(dictionary, old_pickle):
    """
    Concat the files from old pickle with the new files to add.
    """
    for key in dictionary.keys():
        old_pickle[key] = list(chain(old_pickle[key], dictionary[key]))

    return old_pickle
    
    
def keep_new_files(files, old_pickle):
    """
    Returns files from the image directories, which doesn't exist already in the old pickle file. 
    """
    pickle_files = []
    for dic, set_files in old_pickle.items():
        for file in set_files:
            pickle_files.append(file)
            
    new_list = list(set(files) - set(pickle_files))
    print("Number of new files added: ", str(len(new_list)))
    
    if (len(new_list) == 0):
        print("ATTENTION: Path to files is empty or maybe wrong. \nIf file is not empty make sure you preserve the directory convention of 'path_to_directory/{covid, non_covid}/images_files_to_load' \n")
    
    return new_list

def split_train_test(dictioanry):
    """
    Split dictionaty of {patiecnt: files}, to train test and val by a ratio, determined in arguments.
    Splitting is done by patients and not by files in order to keep patients seperatly in train and test sets.
    Split is done with limitation of maximum number of images per patient in test set, determined in arguments.
    First we fill up the (val set - optianal and ) test set up till the desired ratio, then the rest goes to train.  
    """
    train_set = []
    test_set = []
    val_set = []
    
    if (args.first_image): # keep only first image of each patient.
        num_patinent = len(dictioanry)
        for key,value in dictioanry.items():
            if(args.validation and (len(val_set) < args.test_size*num_patinent)): # val set
                val_set.append(value[0]) 
            if (len(test_set) < args.test_size*num_patinent): # test set
                test_set.append(value[0])
            else:    # train set
                train_set.append(value[0])
    
    else: # keep all images of each patient.
        num_images = sum([len(x) for x in dictioanry.values()])
        for key,value in dictioanry.items():
            if(args.validation and (len(val_set) < args.test_size*num_images)): # val set
                val_set = list(chain(val_set, value))    
            elif ((len(test_set) < args.test_size*num_images) and (len(value) <= args.max_images)): # test set
                test_set = list(chain(test_set, value))
            else: # train set
                train_set = list(chain(train_set, value))
                
    if(args.validation):
        return {'train': train_set, 'test': test_set, 'val': val_set}
    else:
        return {'train': train_set, 'test': test_set}
        
def create_patient_dict(file_list):
    """
    Loop over files, find patients ID, and create dictionary of {patient_ID: all_files_of_patient}
    """
    patients_dict = {}
    for file_path in file_list:
        file_id = file_path.split(sep = '/')[-1]       # different ID prefixes of different sources
        if (file_id[:3] in ['SHc', 'Shc']):
            id_length = 6
        elif (file_id[:1] in ['N']):
            id_length = 5
        elif (file_id[:2] in ['be', 'BE', 'ME', 'MM', 'RO', 'sh', 'SH']):
            id_length = 5
        elif (file_id[:2] in ['EB', 'Ee', 'SZ']):
            id_length = 6
        elif (file_id[:4] in ['SZMC']):
            id_length = 6
        elif(file_id.split('_')[0].isdigit() or file_id.split('_')[0].isalpha()):
            s = file_id.split('_')[0]
            id_length = len(s)
        elif(file_id.split('.')[0].isdigit() or file_id.split('.')[0].isalpha()):
            s = file_id.split('.')[0]
            id_length = len(s)
        else:
            s = file_id.split('_')[0]
            id_length = len(s)
            print("Unrecogized file name: ", file_id)
            
        pid = file_id[:id_length] #define id
        if pid not in patients_dict: # if id not yet storred, create new id key
            patients_dict[pid] = [file_path]
        else: # add file to existing id key
            patients_dict[pid].append(file_path)

    return patients_dict

def read_and_shuffle_list(path_to_direcotry, label):
    """
    Read all files recursivle from directory into list.
    Shuffling is made in the list so splitting won't be by ABC or by sources order.
    """
    imgs_pathes = glob.glob(path_to_direcotry + '**/' + label + '/*', recursive = True)
    files_list = [x.split(sep = '/')[-1] for x in imgs_pathes]
    
    random.Random(RANDOM_STATE).shuffle(imgs_pathes)
    
    return imgs_pathes

                             
def split(args):
    """
    Manage the splitting of files to train, test and val (optional)
    Two modes of splitting the files: 'new' create a new pickle of splitted files, 'update' is loading a pickle alonside reading files from directory and adding only new files. The new files are added to pickles split without thus keeping the split of the pickle.
    """
    if(args.update):     
        #load existing pickle
#         with open(PATH_PICKLE, 'rb') as f: 
        with open(args.update, 'rb') as f: 
            old_pickle = pickle.load(f)
            pickle_covid = old_pickle['dict_covid']
            pickle_non_covid = old_pickle['dict_non_covid']

        # read files from input directory
        files_covid_shuffled = read_and_shuffle_list(args.input_folder, 'covid')
        files_non_covid_shuffled = read_and_shuffle_list(args.input_folder, 'non_covid')

        # drop file which already exist in pickle
        files_covid_new = keep_new_files(files_covid_shuffled, pickle_covid)
        files_non_covid_new = keep_new_files(files_non_covid_shuffled, pickle_non_covid)
          
        # create patients ditionary from the files lists
        dict_covid_patients = create_patient_dict(files_covid_new)
        dict_non_covid_patients = create_patient_dict(files_non_covid_new)
        
        # split dictionaries to train test 
        dict_covid_splitted = split_train_test(dict_covid_patients) 
        dict_non_covid_splitted= split_train_test(dict_non_covid_patients)   

        # add the new files to existing pickle and return new train and test sets
        dict_covid = concat_dictionaries(dict_covid_splitted ,pickle_covid)   
        dict_non_covid = concat_dictionaries(dict_non_covid_splitted ,pickle_non_covid)
        
        
    else:
        # read files from input directory
        covid_files_shuffled = read_and_shuffle_list(args.input_folder, 'covid')
        non_covid_files_shuffled = read_and_shuffle_list(args.input_folder, 'non_covid')

        # create patients ditionary from the files lists
        covid_patients_dict = create_patient_dict(covid_files_shuffled)
        non_covid_patients_dict = create_patient_dict(non_covid_files_shuffled)

        # split dictionaries to train test 
        dict_covid = split_train_test(covid_patients_dict) 
        dict_non_covid = split_train_test(non_covid_patients_dict)         
    
    # balane data if requested
    if(args.balance):
        dict_covid, dict_non_covid = balance_data(dict_covid, dict_non_covid)
        
    return dict_covid, dict_non_covid


def move_files_and_save(dict_covid, dict_non_covid):
    """
    Manage the end of process: create symlinks and save documentation
    """
    # create output directory with symlinks
    print("Creating symlinks to files...")
    move_files(dict_covid, 'covid')
    move_files(dict_non_covid, 'non_covid')
    
    # save documentatoin of splitting
    save_list(dict_covid, dict_non_covid)    
    save_pickle(dict_covid, dict_non_covid)
    
    #print results of splitting
    print()
    for key in dict_covid.keys():
        print("len of {key}_covid: ".format(key = key), len(dict_covid[key]))
        print("len of {key}_non_covid: ".format(key = key), len(dict_non_covid[key]))
    
def run(args):
    dict_covid, dict_non_covid = split(args)
    move_files_and_save(dict_covid, dict_non_covid)
    print("\nFinished")

        
    
########################################################################################################################

def fill_parser(parser: ArgumentParser):
    '''
    Use this from external file
    :return:
    '''
    parser.add_argument('input_folder', help='Input data folder to split to train and test')
    parser.add_argument('output_folder', help='Output folder to save splitted data')

    parser.add_argument('--update', help = 'update input folder to existing pickle keeping its divisions. pickle path needs to be provided with argument', action = 'store')
                        
    parser.add_argument('--validation', help = 'create also validation set', dest='validation', action='store_true')
    parser.add_argument('--no-validation', dest='validation', action='store_false')
    parser.set_defaults(validation=False)

    parser.add_argument('--first_image', dest='first_image', action='store_true')
    parser.add_argument('--no-first_image', dest='first_image', action='store_false')
    parser.set_defaults(first_image=False)

    parser.add_argument('--shuffle', dest='shuffle', action='store_true', help = 'use a ramdom seed instead of default seed to shuffle the data')
    parser.add_argument('--no-shuffle', dest='shuffle', action='store_false', help = 'use the default seed (1234) to shuffle the data so it can be restored')
    parser.set_defaults(shuffle=False)

    parser.add_argument('--balance', dest='balance', action='store_true')
    parser.add_argument('--no-balance', dest='balance', action='store_false')
    parser.set_defaults(balance=False)

    parser.add_argument('--test_size', type=float, default=0.15)
    
    parser.add_argument('--max_images', type=int, default=5)

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
    run(args)

if __name__ == '__main__':
    args = get_args()

    # choose new random state if shuffling is desired
    if (args.shuffle):
        RANDOM_STATE = random.randrange(1000)
    #use default random_state
    else:
        RANDOM_STATE = 1234 #default random state

    main(args)
    ### unit test code ###

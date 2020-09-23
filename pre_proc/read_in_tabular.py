import os
import re
import os.path as osp
import pandas as pd
import numpy as np
from argparse import ArgumentParser

import datetime

print(os.getcwd())

from constants import cfg

##### Load constants from constants folder ################
DATA_PATH = cfg.DATA_PATH_TABULAR

##### Helpers ####
def year_two2four_digits(yearstr):
    if len(yearstr) != 2:
        return yearstr

    if int(yearstr) <= 20:
        return '20' + yearstr
    else:
        return '19' + yearstr

def unify_shift_string(shift):
    """
    Get unified shift strings
    For example, M, mor, morn, Morning --> all turn to morning
    """
    if pd.isna(shift):
        return shift

    if type(shift) != str:
        print("got shift", shift, " of type", type(shift), "instead of expected string")
        return shift

    MORNING_STRINGS = ['m', 'm1', 'mo', 'mor', 'mon', 'morn', 'morning']
    EVENING_STRINGS = ['e', 'e1', 'eve', 'evening']
    NIGHT_STRINGS = ['n', 'nig', 'nih', 'nigh', 'night']

    def get_num():
        """
        Sometimes there is more than one image in a shift per person, and then we get a
        number at the end of the shift name (e.g. eve1, eve2) - this function extracts this number
        """
        num = ''
        shift_num = re.findall("[0-9]+", shift)
        if len(shift_num) == 1:
            num = shift_num[0]
        return num

    shift = shift.lower()
    shift_char = re.findall("[a-zA-Z]+", shift)

    if len(shift_char) < 0:
        return np.nan

    shift_char = shift_char[0]

    num = get_num()

    if shift_char in MORNING_STRINGS:
        return 'morning' + num
    elif shift_char in EVENING_STRINGS:
        return 'evening' + num
    elif shift_char in NIGHT_STRINGS:
        return 'night' + num
    elif 'poor' in shift:
        return 'bad' + num
    return 'xxx'

def unify_date(date):
    if type(date) == str:
        return unify_date_string(date)
    elif type(date) == datetime.datetime:
        return unify_date_datetime(date)
    elif pd.isna(date):
        return date
    else:
        print("date of unexpected type: ", date)

def unify_date_datetime(date,separator = "_",month_first=True):
    if month_first:     return str(date.month) + separator + str(date.day) + separator + str(date.year)
    return str(date.day) + separator + str(date.month) + separator + str(date.year)

def unify_date_string(date,separator = "_"):
    """
    Get unified date strings
    For example, 1/13/2019 --> turns to 01/13/19
                 11/22/2020--> turns to 11/22/20
                 12/12/20  --> stays    12/12/20
                 1/1/11    --> turns to 01/01/11
    unify dates to have two/two/four
    """
    if '/' not in date:
        return unify_date_string_no_slash(date, separator)

    date = date.lower()
    date_parts = date.split("/")
    new_date = ''

    for dprt in date_parts[:2]:
        # month and day formating, e.g. : turns 6 to 06
        dprt = dprt.strip(' ')
        if len(dprt) == 1:
            dprt = '0' + dprt
        new_date = new_date + dprt + separator

    year_part = date_parts[-1].strip(' ')
    if len(year_part) == 2:
        year_part = year_two2four_digits(year_part)
    new_date = new_date + year_part

    return new_date

def unify_date_string_no_slash(date, separator = "_"):

    year = date[-4:]
    if not(year.startswith('20') or year.startswith('19')):
        year = year_two2four_digits(date[-2:])
    new_date = date[:2] + separator + date[2:4] + separator + year
    return new_date

def get_age_years(s):
    """
    Get age in years
    for example 14Y 6M --> turns to 14
    """
    if type(s) == int:
        return s
    elif pd.isna(s):
        return s
    a = re.findall('[0-9]*Y', s)
    return int(a[0].strip('Y'))

def to_int(x):
    try:
        return int(x)
    except:
        return x

##### Unify file names #####

def unify_file_names_in_dir(dir):
    """
    :param dir: directory with file names of the format "patientid_date_shift.tiff",
                for example : Ee0004_19032020_eve.tiff
    changes the names of the files in the directory,
     to file names of the same format, but with dates and shifts unified
    """
    for count, file_name in enumerate(os.listdir(dir)):
        new_file_name = unify_file_name(file_name)
        source = os.path.join(dir, file_name)
        destination = os.path.join(dir, new_file_name)
        os.rename(source, destination)

def unify_file_name(name):
    name, ending = name.split('.')
    patient_id, date, shift = name.split('_')
    return '_'.join([patient_id, unify_date_string(date), unify_shift_string(shift)]) + '.' + ending

##### Clean up #####

def filter_low_quality(tabular_df, QUALITY_COL_NAME):
    # Filter rows where quality = 0 ( or even quality = 1, should ask doctors)
    rows2drop = tabular_df.index[tabular_df[QUALITY_COL_NAME] == 0]
    tabular_df.drop(rows2drop, axis = 0, inplace = True)
    return tabular_df

def clean_df(tabular_df):
    tabular_df.drop(tabular_df.index[pd.isna(tabular_df.id)], axis=0, inplace=True)
    # drop rows with nans in multiple other columns
    tabular_df.drop(tabular_df.index[np.logical_and(pd.isna(tabular_df.Age),
                                                    np.logical_and(pd.isna(tabular_df.Mas),
                                                                   np.logical_and(pd.isna(tabular_df.Date),
                                                                                  pd.isna(tabular_df.Gender)
                                                                                  )
                                                                   )
                                                    )
                                    ], axis=0, inplace=True)
    tabular_df.reset_index(drop=True, inplace=True)
    # treat "." values as nans
    tabular_df.replace(".", np.nan, inplace=True)
    return tabular_df

##### Preprocess tables #####

def preprocess_nogah(tabular_df):
    """
    preprocess data frame :
        - make categorical one-hot encoded

    TO DO :
      - Encode the comment
      - is SourceAE the image filename?
      - should Time (hour taken) or shift (morning, night, eve) be a feature?
      - What is Mod and why is it only in Ahuva's data?
    """
    CONVERT_POS_TO_AHUVA_STANDARD = {0 : 3, #Supine
                                     1 : 2, #Sitting
                                     2 : 1} #Standing

    POS_COL_NAME = 'posture (supine=0, sitting=1, standing=2)' # In Ahuva's data it is : 'POSITION:\n1 - Standing\n2 - Sitting\n3 - Supine'    GENDER_COL_NAME = 'Gender'
    QUALITY_COL_NAME = 'quality (0=poor, 1=acceptible, 2=good)'
    # change column names

    tabular_df.columns = [col.lower() for col in tabular_df.columns] # make all columns lower case

    tabular_df = tabular_df.rename({'research no' : 'id'}, axis = 1)
    tabular_df['source'] = 'nogah'

    tabular_df = clean_df(tabular_df)
    tabular_df = filter_low_quality(tabular_df, QUALITY_COL_NAME)
    tabular_df["Date"] = tabular_df["Date"].apply(unify_date_string)
    tabular_df["Shift"] = tabular_df["Shift"].apply(unify_shift_string)

    tabular_df['filename'] = tabular_df['id'] + "_" + tabular_df['Date'] + "_" + tabular_df["Shift"]

    # retrieve numerical age in years from Y, M string age format
    tabular_df['Age'] = tabular_df['Age'].apply(get_age_years)

    tabular_df[POS_COL_NAME] = tabular_df[POS_COL_NAME].replace("*1", "1")# treat *1 as 1
    # convert position to the standard used in Ahuva's data
    tabular_df[POS_COL_NAME] = tabular_df[POS_COL_NAME].apply(lambda x : CONVERT_POS_TO_AHUVA_STANDARD[x])

    pos_df = pd.get_dummies(tabular_df[POS_COL_NAME], prefix='position', drop_first=True, dummy_na=True)
    #shift_df = pd.get_dummies(tabular_df['Shift'], prefix='Shift', drop_first=True, dummy_na=True)

    new_df = pd.concat([tabular_df[['id', 'filename', 'age', 'gender', 'mas', 'kvp', 'source']], pos_df], axis=1)
    new_df = preprocess_general(new_df)
    return new_df

def unify_column_names(df, terms):
    """
    takes all columns
    """
    for term in terms:
        for col in df.columns:
            if term in col.lower():
                df = df.rename({col : term})
    return df

def preprocess_ahuva(tabular_df):
    """
    preprocess data frame :
        - make categorical one-hot encoded, including category for nan
        - leave continuous (age) as-is
        - assumes no nan in gender / age
        - doesn't return the date column
    """
    POS_COL_NAME = 'POSITION:\n1 - Standing\n2 - Sitting\n3 - Supine'
    GENDER_COL_NAME = 'GENDER'
    AGE_COL_NAME = 'AGE (Y)'

    tabular_df.columns = [col.lower() for col in tabular_df.columns] # make all columns lower case

    tabular_df = unify_column_names(tabular_df, ['position', 'serialn'])

    # change column names
    tabular_df = tabular_df.rename({GENDER_COL_NAME : 'gender',
                                    AGE_COL_NAME : 'age',
                                    'serialn' : 'id',
                                    'file name' : 'filename',
                                    'kv' : 'kvp'}, axis = 1)

    # add 10K to each id in Ahuva's group, so we get a separation between Ahuva's and Nogah's database
    tabular_df['id'] = tabular_df['id'].apply(lambda x: x + 10000)
    tabular_df = clean_df(tabular_df)
    tabular_df['source'] = 'ahuva'

    tabular_df[POS_COL_NAME] = tabular_df[POS_COL_NAME].replace("*1", "1")    # treat *1 as 1

    pos_df = pd.get_dummies(tabular_df[POS_COL_NAME], prefix='position', drop_first=True, dummy_na=True)

    #MACHINE_COL_NAME = 'MACHINE: \n1 - Siemens\n2 - Shimadzu\n3 - Samsung\n4 - Fuji\n5 - Carestream'
    #MOD_COL_NAME = 'MOD.'
    #machine_df = pd.get_dummies(tabular_df[MACHINE_COL_NAME], prefix='machine', drop_first=True, dummy_na=True)
    #mod_df = pd.get_dummies(tabular_df[MOD_COL_NAME], prefix='MOD', drop_first=True, dummy_na=True)

    new_df = pd.concat([tabular_df[['id', 'filename', 'age', 'gender', 'mas', 'kvp', 'source']], pos_df], axis=1)
    new_df = preprocess_general(new_df)

    return new_df

def preprocess_general(tabular_df):
    """
    preprocess functions that are the same no matter what the source is
    """
    # change gender to binary
    tabular_df["Gender"] = tabular_df["Gender"].apply(lambda  x: int(x == 'F'))

    # fill in NaNs in Mas as 3, NaNs in KvP as 85
    tabular_df.loc[pd.isna(tabular_df['Mas']), 'Mas'] = 3
    tabular_df.loc[pd.isna(tabular_df['KvP']), 'KvP'] = 85

    tabular_df['Mas'] = tabular_df['Mas'].apply(to_int)
    tabular_df['KvP'] = tabular_df['KvP'].apply(to_int)
    return tabular_df

def preprocess(df, source):
    if source == 'ahuva':
        return preprocess_ahuva(df)
    elif source == 'nogah':
        return preprocess_nogah(df)
    else:
        print("unknown source", source, "is neither ahuva nor nogah")

##### Merge them together #####

def add_new_tabular(curr_file_path, new_file_name, source):
    """
    :param curr_file_path: path to the folder with the current tabular file - this file contains all the data gathered so far
    and is called nogah.csv or ahuva.csv depending on the source
    :param new_file_name: new tabular file name for the new data file - where to read it from
    :param source: 'nogah' or 'ahuva'
    """
    curr_file = pd.read_csv(curr_file_path + source + '.csv')
    new_file = pd.read_csv(new_file_name)
    new_file = preprocess(df=new_file, source=source)

    merged_file = pd.concat([curr_file, new_file])
    merged_file.drop_duplicates(keep='last', inplace=True)
    return merged_file

def read_in_table(path):
    """
    reads in table, supports csv and xlsx formats
    """
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".xlsx"):
        return pd.read_excel(path)

def save_table(table, path):
    if path.endswith(".csv"):
        table.to_csv(path)
    elif path.endswith(".xlsx"):
        table.to_excel(path)

def get_tables(path_list, source_list):
    new_tables = []
    for table_path, table_source in zip(path_list, source_list):
        table = read_in_table(table_path)
        table = preprocess(table, table_source)
        new_tables.append(table)
    return new_tables

def merge_tables(tables):
    """
    merge a list of tables (assumed to have the same columns)
    """
    merged_file = pd.concat(tables)
    merged_file.drop_duplicates(keep='last', inplace=True)
    return merged_file

def merge_tables_and_save(path_list, source_list, save_path):
    """
    takes a path, and merges all the tables of the sources in this path
    (assumed to have the same columns)
    and save to a new file
    """
    new_tables  = get_tables(path_list, source_list)
    merged_file = merge_tables(new_tables)
    merged_file.to_csv(save_path)

def add_iter_to_name(filename, iter):
    """
    add the augmentation iteration identifier - i.e. A, B, C, D
    to the end of the image file name
    """
    filename_split = filename.split(".")
    if len(filename_split) == 1: # no ending
        filename, ending = filename_split[0], ""
    elif len(filename_split) == 2: # with ending
        filename, ending = filename_split
        ending =  "." + ending
    else: # multiple dots - odd
        print("got error in filename, misplaced dot", filename)
        filename, ending = filename_split[0], filename_split[-1]
        ending =  "." + ending
    return filename + iter + ending

def duplicate_augmented_rows(df, filename_column = "filename", augmentation_itertions = ["A","B","C","D"]):
    """
    takes a dataframe and duplicates the rows according to the augmentations that were done
    """
    reps = len(augmentation_itertions)
    newdf = pd.DataFrame(np.repeat(df.values, reps, axis=0))
    newdf.columns = df.columns

    orig_file_names = df[filename_column]
    new_file_names = [""]*(len(orig_file_names)*reps)
    for file_idx, filename in enumerate(orig_file_names):
        for rep_idx, iter in enumerate(augmentation_itertions):
            new_file_names[reps*file_idx + rep_idx] = add_iter_to_name(filename, iter)

    newdf[filename_column] = new_file_names
    return newdf

def fill_parser(parser: ArgumentParser):
    '''
    Use this from external file
    :return:
    '''
    parser.add_argument('-sources', nargs='+', default=['ahuva', 'nogah'],
                        help='Source csv names, e.g. -sources ahuva nogah')
    parser.add_argument('-tablepaths', nargs='+', default=['ahuva.csv', 'nogah.csv'],
                        help='csv full paths, e.g. -sources ahuva nogah')
    parser.add_argument('-outdir', nargs='+', default=['/'],
                        help='csv full paths, e.g. -sources ahuva nogah')
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
    """
    main entry point to script, from external calls as well as __main__
    """
    path_list = [tpath for tpath in args.tablepaths ]

    save_path = osp.join(args.outdir, 'merged_table.csv')

    merge_tables_and_save(path_list=path_list,
                          source_list=args.sources,
                          save_path=save_path)

    tabular_dfs = {}

    for tablepath, source in zip(args.tablepaths, args.sources):
        tabular_dfs[source] = read_in_table(tablepath)

if __name__ == '__main__':
    args = get_args()
    main(args)




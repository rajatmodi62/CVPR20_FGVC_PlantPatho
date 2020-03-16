import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from numpy.random.mtrand import RandomState
from sklearn.model_selection import StratifiedKFold

from utils.config import load_config

CLASSES = {'road,parking,drivable fallback': 0,
           'sidewalk,rail track,non-drivable fallback': 1,
           'person,animal,rider': 2,
           'motorcycle,bicycle,autorickshaw,car,bus,truck,caravan,trailer,train,vehicle': 3,
           'curb,wall,fence,guard rail, traffic light, sign, pole': 4,
           'building,bridge,tunnel,vegetation': 5,
           'sky,fallback background': 6,
           'unlabeled, ego vehicle, rectification border, out of roi, license plate': 255,
           }

'''label- classes containing the class information
group_column- ImageId columns which need to be properly grouped together.
 '''
def stratified_group_k_fold(
        label: str,
        group_column: str,
        df: pd.DataFrame = None,
        file: str = None,
        n_splits=5,
        seed: int = 0
):
    random_state = RandomState(seed)

    if file is not None:
        df = pd.read_csv(file)

    #create a dictionary containing image ids and their label sets

    labels = defaultdict(set)
    for g, l in zip(df[group_column], df[label]):
        labels[g].add(l)


    #we have an imageid and a set of classes.
    group_labels = dict()
    groups = []
    Y = []
    for k, v in labels.items():
        group_labels[k] = random_state.choice(list(v))
        Y.append(group_labels[k])
        groups.append(k)

    index = np.arange(len(group_labels))
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True,
                            random_state=random_state).split(index, Y)
    group_folds = dict()

    #stratified kfold ensures that each image will go into val set once.
    for i, (train, val) in enumerate(folds):
        for j in val:
            group_folds[groups[j]] = i


    res = np.zeros(len(df))
    for i, g in enumerate(df[group_column]):
        res[i] = group_folds[g]

    return res.astype(np.int)

    def stratified_k_fold(
            label: str, df: pd.DataFrame = None, file: str = None, n_splits=5,
            seed: int = 0
    ):
        random_state = RandomState(seed)

        if file is not None:
            df = pd.read_csv(file)

        index = np.arange(df.shape[0])
        res = np.zeros(index.shape)
        folds = StratifiedKFold(n_splits=n_splits,
                                random_state=random_state,
                                shuffle=True).split(index, df[label])

        for i, (train, val) in enumerate(folds):
            res[val] = i
        return res.astype(np.int)

#define class 7 for the masks which are empty
def split_folds(config_file):
    config = load_config(config_file)
    os.makedirs(config.work_dir, exist_ok=True)
    df = pd.read_csv(config.data.train_df_path)
    df['exists']=df['EncodedPixels'].notnull().astype(int)
    df['Fold_ClassId']=[row.ClassId if row.exists else 7 for row in df.itertuples()]
    df['fold'] = stratified_group_k_fold(
        label='Fold_ClassId', group_column='ImageId', df=df, n_splits=config.data.params.num_folds)

    #rearrange the dataframe on the axis
    #imageid as index, classes as rle encoding and bagging fold no
    pv_df = df.pivot(index='ImageId', columns='ClassId',
                     values='EncodedPixels')
    pv_df = pv_df.merge(df[['ImageId', 'fold']], on='ImageId', how='left')
    pv_df = pv_df.drop_duplicates()
    pv_df = pv_df.set_index('ImageId')
    pv_df.to_csv('dataset/train_folds.csv')
    print("success")

#argparser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_file',
                        help='configuration file path',
                        default=None, type=str)
    return parser.parse_args()

#split the training data into folds for bagging

def main():
    print('split train dataset')
    args = parse_args()
    if args.config_file is None:
        raise Exception('no configuration file')
    print('load config from {}'.format(args.config_file))
    split_folds(args.config_file)


if __name__ == '__main__':
    main()

import os
import sys
import torch
import albumentations as A
from retinanet.detection.dataset.csvdataset import get_train_transforms, get_val_transforms, DatasetRetriever
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import torchvision.transforms as transforms
import torchvision.datasets as datasets


class Config(object):
    log = './log'  # Path to save log
    checkpoint_path = './checkpoints'  # Path to store checkpoint model
    resume = './checkpoints/latest.pth'  # load checkpoint model
    evaluate = None  # evaluate model path

    network = "resnet50_retinanet"
    pretrained = False
    num_classes = 1
    seed = 0
    input_image_size = 640

    marking = pd.read_csv('../input/global-wheat-detection/train.csv')

    bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))  # shape(147793,4)
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        marking[column] = bboxs[:, i]  # 相当于在train_df表格中又加入了四列  box的xywh
    marking.drop(columns=['bbox'], inplace=True)  # 之后把Bbox列丢掉，因为已经加入了xywh

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    df_folds = marking[['image_id']].copy()  # df变成一列表格，title是image_id, 内容是图片名
    df_folds.loc[:, 'bbox_count'] = 1  # 给df_folds加一列，名叫Bbox count 初始化所有值为1
    df_folds = df_folds.groupby('image_id').count()  # 计算每个图片的框框数
    df_folds.loc[:, 'source'] = marking[['image_id', 'source']].groupby('image_id').min()['source']
    df_folds.loc[:, 'stratify_group'] = np.char.add(
        df_folds['source'].values.astype(str),
        df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
    )
    df_folds.loc[:, 'fold'] = 0
    for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
        df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number
    '''
    最后得到（img num, 5)的矩阵
    列名为 [ image_id, bbox_count, source, stratify_group, fold]
    stratify_group 为分组的依据，表示的是：按一张图上框框的数量， 对图像进行分层，总共分为5层，再用StratifiedKFold按比例分为5个fold
    fold 为分组之后的5组编号
    '''

    train_dataset = DatasetRetriever(
        image_ids=df_folds[df_folds['fold'] != fold_number].index.values,  # 0组设为val组，其他是train组，这里img-ids就是ndarray的所有图片名
        marking=marking,
        transforms=get_train_transforms(),
        test=False,
    )

    validation_dataset = DatasetRetriever(
        image_ids=df_folds[df_folds['fold'] == fold_number].index.values,
        marking=marking,
        transforms=get_val_transforms(),
        test=True,
    )


    epochs = 12
    batch_size = 4
    lr = 1e-4
    num_workers = 1
    print_interval = 100
    apex = False
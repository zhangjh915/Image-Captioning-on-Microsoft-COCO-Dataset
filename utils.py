# This code is modified from CS231n.
import os
import json
import numpy as np
import h5py

DATA_DIR = 'data/coco_captioning'  # define the dataset path


def load_coco_dataset(data_dir=DATA_DIR, PCA_features=True):
    data = {}
    caption_file = os.path.join(data_dir, 'coco2014_captions.h5')
    with h5py.File(caption_file, 'r') as f:  # read caption file with h5py
        for k, v in f.items():
            data[k] = np.asarray(v)

    # extract training features
    if PCA_features:
        train_feature_file = os.path.join(data_dir, 'train2014_vgg16_fc7_pca.h5')
    else:
        train_feature_file = os.path.join(data_dir, 'train2014_vgg16_fc7.h5')
    with h5py.File(train_feature_file, 'r') as f:
        data['train_features'] = np.asarray(f['features'])

    # extract validation features
    if PCA_features:
        val_feature_file = os.path.join(data_dir, 'val2014_vgg16_fc7_pca.h5')
    else:
        val_feature_file = os.path.join(data_dir, 'val2014_vgg16_fc7.h5')
    with h5py.File(val_feature_file, 'r') as f:
        data['val_features'] = np.asarray(f['features'])

    # extract index-to-word and word-to-index into dictionary
    dict_file = os.path.join(data_dir, 'coco2014_vocab.json')
    with open(dict_file, 'r') as f:
        dict_data = json.load(f)
        for k, v in dict_data.items():
            data[k] = v

    # read image files from website, note that some of them might not be available for now
    train_url_file = os.path.join(data_dir, 'train2014_urls.txt')  # this file includes urls for the training images
    with open(train_url_file, 'r') as f:
        train_urls = np.asarray([line.strip() for line in f])
    data['train_urls'] = train_urls

    val_url_file = os.path.join(data_dir, 'val2014_urls.txt')  # this file includes urls for the validation images
    with open(val_url_file, 'r') as f:
        val_urls = np.asarray([line.strip() for line in f])
    data['val_urls'] = val_urls

    return data


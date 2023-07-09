# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json


class VideoData(Dataset):
    def __init__(self, mode, video_type, split_index=None, pretrain=False):
        """ Custom Dataset class wrapper for loading the frame features and ground truth importance scores.

        :param str mode: The mode of the model, train or test.
        :param str video_type: The Dataset being used, SumMe or TVSum.
        :param int split_index: The index of the Dataset split being used.
        """
        self.mode = mode
        self.name = video_type.lower()
        self.datasets = ['../PGL-SUM/data/datasets/SumMe/eccv16_dataset_summe_google_pool5.h5',
                         '../PGL-SUM/data/datasets/TVSum/eccv16_dataset_tvsum_google_pool5.h5',
                         '/media/external_10TB/10TB/p_haghighi/summarization/CLIP_guided_dataset/output_merge.h5']
        self.splits_filename = ['../PGL-SUM/data/datasets/splits/' + self.name + '_splits.json']
        self.split_index = split_index  # it represents the current split (varies from 0 to 4)
        self.pretrain = pretrain

        if self.pretrain:
            self.filename = self.datasets[2]  # Use ActivityNet for pretraining
            hdf = h5py.File(self.filename, 'r')
            self.list_frame_features, self.list_gtscores = [], []
            for video_name in hdf.keys():
                frame_features = torch.Tensor(np.array(hdf[video_name + '/features']))
                gtscore = torch.Tensor(np.array(hdf[video_name + '/gtscore']))
                self.list_frame_features.append(frame_features)
                self.list_gtscores.append(gtscore)
            hdf.close()
        else:
            if 'summe' in self.splits_filename[0]:
                self.filename = self.datasets[0]
            elif 'tvsum' in self.splits_filename[0]:
                self.filename = self.datasets[1]

            hdf = h5py.File(self.filename, 'r')
            self.list_frame_features, self.list_gtscores = [], []

            with open(self.splits_filename[0]) as f:
                data = json.loads(f.read())
                for i, split in enumerate(data):
                    if i == self.split_index:
                        self.split = split
                        break

            for video_name in self.split[self.mode + '_keys']:
                frame_features = torch.Tensor(np.array(hdf[video_name + '/features']))
                gtscore = torch.Tensor(np.array(hdf[video_name + '/gtscore']))

                self.list_frame_features.append(frame_features)
                self.list_gtscores.append(gtscore)

            hdf.close()

    def __len__(self):
        """ Function to be called for the `len` operator of `VideoData` Dataset. """
        if self.pretrain:
            self.len = len(self.list_frame_features)
        else:
            self.len = len(self.split[self.mode+'_keys'])
        return self.len

    def __getitem__(self, index):
        """ Function to be called for the index operator of `VideoData` Dataset.
        train mode returns: frame_features and gtscores
        test mode returns: frame_features and video name

        :param int index: The above-mentioned id of the data.
        """
        if self.pretrain:
            frame_features = self.list_frame_features[index]
            gtscore = self.list_gtscores[index]

            gtscore -= gtscore.min()
            if gtscore.max() > 0:
                gtscore /= gtscore.max()

            return frame_features, gtscore
        else:
            video_name = self.split[self.mode + '_keys'][index]
            frame_features = self.list_frame_features[index]
            gtscore = self.list_gtscores[index]

            if self.mode == 'test':
                return frame_features, video_name
            else:
                return frame_features, gtscore


def get_loader(mode, video_type, split_index=None, pretrain=False):
    """ Loads the `data.Dataset` of the `split_index` split for the `video_type` Dataset.
    Wrapped by a Dataloader, shuffled and `batch_size` = 1 in train `mode`.

    :param str mode: The mode of the model, train or test.
    :param str video_type: The Dataset being used, SumMe or TVSum.
    :param int split_index: The index of the Dataset split being used.
    :return: The Dataset used in each mode.
    """
    if mode.lower() == 'train' or pretrain:
        vd = VideoData(mode, video_type, split_index, pretrain)
        return DataLoader(vd, batch_size=1, shuffle=True)
    else:
        return VideoData(mode, video_type, split_index)


if __name__ == '__main__':
    pass

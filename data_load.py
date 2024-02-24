import os
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

def zeros():
    return [0,0]


def train_dataloader(data_path, file_name, class_num, batch_size=32, num_workers=0):

    dataloader = DataLoader(
        RecordDataset(data_path, 'record.total', file_name, class_num),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True
    )
    return dataloader


def test_dataloader(data_path, file_name, class_num, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        RecordDataset(data_path, 'record.total', file_name, class_num),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader



def valid_dataloader(data_path, file_name, class_num, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        RecordDataset(data_path, 'record.total', file_name, class_num),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


class RecordDataset(Dataset):
    def __init__(self, data_path, mode, file_name, class_num, is_test=False):
        self.data_path = data_path
        self.file_path = os.path.join(data_path, mode + '.' + file_name)
        self.is_test = is_test

        data = np.load(self.file_path)
        self.record_id = data['arr_0']
        self.user_id = data['arr_1']
        self.answer_set = data['arr_2']
        self.mask_matrix = data['arr_3']
        self.priori_matrix = data['arr_4']
        self.class_num = class_num
        self.cumulative_priori = defaultdict(zeros)

    def __len__(self):
        return self.answer_set.shape[0]

    def __getitem__(self, idx):
        record_id = self.record_id[idx]
        user_id = self.user_id[idx]
        answer = self.answer_set[idx]
        mask = self.mask_matrix[idx]
        priori = self.priori_matrix[idx]

        answer = torch.from_numpy(answer.copy()).float()
        mask = torch.from_numpy(mask.copy()).float()

        self.cumulative_priori[record_id] = self.cumulative_priori[record_id] + priori

        return record_id, user_id, answer, mask, self.cumulative_priori


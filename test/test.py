from torch.utils.data.dataset import Dataset
import pandas as pd
import os

from typing import Set


class FeatDataset(Dataset):
    def __init__(self, file_path, sets, **kwargs):
        super(FeatDataset, self).__init__()

        # 读取文件: csv存储(, file_path, length, label)
        self.root = file_path

        # 读取数据集信息
        tables = [pd.read_csv(os.path.join(file_path, s + '.csv')) for s in sets]

        self.table = pd.concat(tables, ignore_index=True).sort_values(by=['length'], ascending=False)

        # 每个数据的路径信息
        X = self.table['file_path'].tolist()

        # 每个数据的长度信息
        X_lens = self.table['length'].tolist()

        # 总计的样本数量
        self.num_samples = len(X)
        print('[Dataset] - Number of individual training instances:', self.num_samples)

        # 使用bucket允许在运行时存在不同的batch size
        self.X = []
        batch_x, batch_len = [], []

        for x, x_len in zip(X, X_lens):
            batch_x.append(x)
            batch_len.append(x_len)
            print(x, "#####", x_len)

            # # Fill in batch_x until batch is full
            # if len(batch_x) == bucket_size:
            #     # Half the batch size if seq too long
            #     if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME) and self.sample_length == 0:
            #         self.X.append(batch_x[:bucket_size // 2])
            #         self.X.append(batch_x[bucket_size // 2:])
            #     else:
            #         self.X.append(batch_x)
            #     batch_x, batch_len = [], []


FeatDataset(file_path='../data/len_for_bucket', sets=['train-clean-100', 'train-clean-360'])
import torch
import numpy as np


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # random sample num_class indices随机打乱活得数字学历, e.g. 5
            for c in classes:
                l = self.m_ind[c]  # all data indices of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch

# 多粒度样本抽取方法
class CategoriesSampler_Hierarchial():

    def __init__(self, label, coarse_label, n_batch, n_cls, n_per): # label, coarse_label都是一维列表,共36000张照片，故两个列表都是36000长度
        self.n_batch = n_batch  # the number of iterations in the dataloader 批次
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)                         # all data label
        coarse_label = np.array(coarse_label)           # all data coarse_label kim
        self.m_ind = []                                 # the data index of each class
        for i in range(max(label) + 1):                 # 0,1,2,...max(label) 每个label都不同，标签最大值就label的总数，59
            # 查找label里面等于i(第i类）的所有图片的下标
            ind = np.argwhere(label == i).reshape(-1)   # 表示所有label=i的元素索引列表,Fc100有100个类，每个类600张图片，所以每次ind有600个 [600]
            ind = torch.from_numpy(ind)                 # 方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。
            self.m_ind.append(ind)                      # 一类一个张量，最后合并在m_ind

    def __len__(self):
        return self.n_batch # 多少批次

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # 随机打乱所有类集合序号，并选择前self.n_cls个类序号，若为5way,则选择5个类的序号
            for c in classes:  # 例如[4,9,1,10,15]
                l = self.m_ind[c]                                    # all data indices of this class
                pos = torch.randperm(len(l))[:self.n_per]            # 随机抽取”类c“的前”self.n_per“个样本（1+15=16张图片）
                batch.append(l[pos])
                # i_batch->batch:包含一个训练批次所选择的训练样本集合
                # 5个tensor：每个tensor表示一个类        每个tensor都有16个图片的下标，根据这个下标可以得到该图片的粗类和细类
                # [ tensor([2493, 2927, 2446, 2766, 2932, 2634, 2720, 2570, 2694, 2747, 2436, 2979, 2607, 2501, 2525,2558]),
                #   tensor([5778, 5730, 5442, 5740, 5988, 5858, 5973, 5780, 5413, 5969, 5834, 5670, 5716, 5638, 5773,5672]),
                #   tensor([ 702,  897,  785, 1186, 1025,  758,  625,  975,  807,  723,  931,  769, 1004,  838, 1124,736]),
                #   tensor([6315, 6146, 6358, 6596, 6019, 6216, 6558, 6111, 6236, 6157, 6307, 6063, 6229, 6257, 6159,6301]),
                #   tensor([9325, 9472, 9244, 9291, 9451, 9258, 9066, 9279, 9386, 9095, 9401, 9579, 9575, 9434, 9240,9501]) ]

            batch = torch.stack(batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch


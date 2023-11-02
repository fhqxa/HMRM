import glob
import os
import os.path as osp

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FC_100(Dataset):

    def __init__(self, setname, args=None, return_path=False):
        TRAIN_PATH = osp.join(args.data_dir, 'FC100/train')
        VAL_PATH = osp.join(args.data_dir, 'FC100/val')
        TEST_PATH = osp.join(args.data_dir, 'FC100/test')
        if setname == 'train':
            THE_PATH = TRAIN_PATH
        elif setname == 'test':
            THE_PATH = TEST_PATH
        elif setname == 'val':
            THE_PATH = VAL_PATH
        else:
            raise ValueError('Wrong setname.')
        data = []
        label = []
        coarse_label = []
        # folders = [osp.join(THE_PATH, label) for label in os.listdir(THE_PATH) if
        #            os.path.isdir(osp.join(THE_PATH, label))]
        # metatrain_folders = [os.path.join(train_folder, coarse_label + '/', label) \
        #                      for coarse_label in os.listdir(train_folder) \
        #                      if os.path.isdir(os.path.join(train_folder, coarse_label)) \
        #                      for label in os.listdir(os.path.join(train_folder, coarse_label))
        #                      ]

        # folders = [osp.join(THE_PATH, coarse_label, label)  \
        #            for coarse_label in os.listdir(THE_PATH) \
        #            if os.path.isdir(os.path.join(THE_PATH, coarse_label)) \
        #            for label in os.listdir(os.path.join(THE_PATH, coarse_label))
        #            ]

        coarse_folders = [osp.join(THE_PATH, coarse_label) for coarse_label in os.listdir(THE_PATH)
                          if os.path.isdir(osp.join(THE_PATH, coarse_label))]
        # 12个粗类
        # ['datasets\\FC100/train\\Fish', 'datasets\\FC100/train\\Flower', 'datasets\\FC100/train\\Food_container', 'datasets\\FC100/train\\Fruit_vegetable',
        # 'datasets\\FC100/train\\Household_electrical_device', 'datasets\\FC100/train\\Household_furniture', 'datasets\\FC100/train\\Large_manmade_outdoor_things',
        # 'datasets\\FC100/train\\Large_natural_outdoor_scene', 'datasets\\FC100/train\\Reptiles', 'datasets\\FC100/train\\Tree', 'datasets\\FC100/train\\Vechicle1', 'datasets\\FC100/train\\Vechicle2']

        '''
            datasets\\FC100/train:
                12个粗类、60个细类         coarse_label:0~11,  label:0~59
                每个细类有600张图片，共36000张图片
        '''

        len_of_coarse = {}#每个粗类的长度
        len_of_coarse[0] = 0
        current_label_index = 0
        for coarse_index in range(len(coarse_folders)):
            fine_folders = [osp.join(coarse_folders[coarse_index], fine_label) for fine_label in os.listdir(coarse_folders[coarse_index]) if
                                   os.path.isdir(osp.join(coarse_folders[coarse_index], fine_label))]
            fine_folders.sort()
            len_of_coarse[coarse_index + 1] = len(glob.glob(os.path.join(coarse_folders[coarse_index], '*/'))) # 获取当前粗节点下的细类数量
            current_label_index += len_of_coarse[coarse_index]
            for idx in range(len(fine_folders)):
                this_folder = fine_folders[idx]
                this_folder_images = os.listdir(this_folder)
                this_folder_images.sort()
                for image_path in this_folder_images:
                    data.append(osp.join(this_folder, image_path))
                    label.append(idx + current_label_index)
                    coarse_label.append(coarse_index)

        self.data = data
        self.label = label
        self.coarse_label = coarse_label
        self.num_class = len(set(label))
        self.return_path = return_path

        # Transformation
        if setname == 'val' or setname == 'test':

            image_size = 84
            resize_size = 92

            self.transform = transforms.Compose([
                transforms.Resize([resize_size, resize_size]),
                transforms.CenterCrop(image_size),

                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        elif setname == 'train':

            image_size = 84

            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label, coarse_label = self.data[i], self.label[i], self.coarse_label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        if self.return_path:
            return image, label, coarse_label, path
        else:
            return image, label, coarse_label


if __name__ == '__main__':
    pass




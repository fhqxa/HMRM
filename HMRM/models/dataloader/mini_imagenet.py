import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np


class MiniImageNet(Dataset):

    def __init__(self, setname, args, return_path=False):
        # args.data_dir = E:/00_Projects_Station/dataset
        TRAIN_PATH = osp.join(args.data_dir, 'miniImagenet/train')
        VAL_PATH = osp.join(args.data_dir, 'miniImagenet/val')
        TEST_PATH = osp.join(args.data_dir, 'miniImagenet/test')
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
        folders = [osp.join(THE_PATH, label) for label in os.listdir(THE_PATH) if
                   os.path.isdir(osp.join(THE_PATH, label))]
        folders.sort()

        for idx in range(len(folders)):
            this_folder = folders[idx]
            this_folder_images = os.listdir(this_folder)
            this_folder_images.sort()
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        self.data = data
        self.label = label
        self.num_class = len(set(label))
        self.return_path = return_path

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
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        if self.return_path:
            return image, label, path
        else:
            return image, label


if __name__ == '__main__':
    pass


# class MiniImageNet(Dataset):
#
#     def __init__(self, setname, args, return_path=False):
#         IMAGE_PATH = os.path.join(args.data_dir, 'miniimagenet/images')
#         SPLIT_PATH = os.path.join(args.data_dir, 'miniimagenet/split')
#
#         csv_path = osp.join(SPLIT_PATH, setname + '.csv')
#         lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
#
#         data = []
#         label = []
#         lb = -1
#
#         self.wnids = []
#         self.args = args
#
#         for l in lines:
#             name, wnid = l.split(',')
#             path = osp.join(IMAGE_PATH, name)
#             if wnid not in self.wnids:
#                 self.wnids.append(wnid)
#                 lb += 1
#             data.append(path)
#             label.append(lb)
#
#         self.data = data  # data path of all data
#         self.label = label  # label of all data
#         self.num_class = len(set(label))
#         self.return_path = return_path
#
#         if setname == 'val' or setname == 'test':
#
#             image_size = 84
#             resize_size = 92
#
#             self.transform = transforms.Compose([
#                 transforms.Resize([resize_size, resize_size]),
#                 transforms.CenterCrop(image_size),
#
#                 transforms.ToTensor(),
#                 transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
#                                      np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
#         elif setname == 'train':
#             image_size = 84
#             self.transform = transforms.Compose([
#                 transforms.RandomResizedCrop(image_size),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
#                                      np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, i):
#         path, label = self.data[i], self.label[i]
#         image = self.transform(Image.open(path).convert('RGB'))
#         if self.return_path:
#             return image, label, path
#         else:
#             return image, label
#
# if __name__ == '__main__':
#     pass


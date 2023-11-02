from common.utils import set_seed


def dataset_builder(args):
    '''
        创建数据集对象
    '''

    set_seed(args.seed)  # fix random seed for reproducibility

    if args.dataset == 'miniimagenet':
        from models.dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'cub':
        from models.dataloader.cub import CUB as Dataset
    elif args.dataset == 'tieredimagenet':
        from models.dataloader.tiered_imagenet import tieredImageNet as Dataset
    elif args.dataset == 'cifar_fs':
        from models.dataloader.cifar_fs import DatasetLoader as Dataset
    elif args.dataset == 'fc100':
        from models.dataloader.fc100 import FC_100 as Dataset
    else:
        raise ValueError('Unkown Dataset')
    return Dataset

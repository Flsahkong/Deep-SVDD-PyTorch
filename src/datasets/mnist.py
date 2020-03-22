from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import MNIST
from src.base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization

import torchvision.transforms as transforms


class MNIST_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier

        # 由于mnist总共有十个种类,所以这里就设置3为正常的normal-class,而其他的都是outlier-class
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)

        # Pre-computed min and max values (after applying GCN) from train data per class
        min_max = [(-0.8826567065619495, 9.001545489292527),
                   (-0.6661464580883915, 20.108062262467364),
                   (-0.7820454743183202, 11.665100841080346),
                   (-0.7645772083211267, 12.895051191467457),
                   (-0.7253923114302238, 12.683235701611533),
                   (-0.7698501867861425, 13.103278415430502),
                   (-0.778418217980696, 10.457837397569108),
                   (-0.7129780970522351, 12.057777597673047),
                   (-0.8280402650205075, 10.581538445782988),
                   (-0.7369959242164307, 10.697039838804978)]

        # GCN就是global contrast normalization
        # MNIST preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        # 将多个归一化的方法组合起来使用
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[normal_class][0]],
                                                             [min_max[normal_class][1] - min_max[normal_class][0]])])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        # 这个类应该是会检测是否有数据集,没有的话会自己下载
        train_set = MyMNIST(root=self.root, train=True, download=True,
                            transform=transform, target_transform=target_transform)
        # 这个train-set.train-labels就是train数据对应的labels,下面这句代码是得到数据3这个label对应的index
        # Subset train_set to normal class
        train_idx_normal = get_target_label_idx(train_set.train_labels.clone().data.cpu().numpy(), self.normal_classes)

        # 重点之中的重点!!!
        # 这里进行了一个subset操作,经过train的数据验证,可以证明,这里的操作是的进行train的数据只有label是3的,而label不是3的则没有进行训练
        # 但是test的时候,没有进行subset,进行test的是所有的数据.
        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MyMNIST(root=self.root, train=False, download=True,
                                transform=transform, target_transform=target_transform)


class MyMNIST(MNIST):
    """Torchvision MNIST class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(MyMNIST, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # 这里返回的这个index是在test的时候,for data in dataloader后面,inputs, labels, idx = data那里用到的
        return img, target, index  # only line changed

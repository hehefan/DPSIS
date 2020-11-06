from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

from .vision import VisionDataset
from .utils import check_integrity, download_and_extract_archive

import scipy.io
import xml.etree.ElementTree

class Dogs(VisionDataset):
    """`Stanford Dataset <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``stanford-dogs`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'stanford-dogs'
    url = "http://vision.stanford.edu/aditya86/ImageNetDogs/"
    filenames = ["images.tar", "annotation.tar", "lists.tar"]
    tgz_md5s = ['1bb1f2a596ae7057f99d7d75860002ef', '4298cc0031f6bc6e74612ac83b5988e2', 'edbb9f16854ec66506b5f09b583e0656']
    file_list = [
        ['file_list.mat', 'ce0676c9520e4b7e1d43221cd9c647b0'],
        ['train_list.mat','d37f459eacccfa4d299373dffba9648d'],
        ['test_list.mat', '66f60c285efbc3ce2fb7893bd26c6b80'],
    ]

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):

        super(Dogs, self).__init__(root)
        self.transform = transform
        self.target_transform = target_transform

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            data = scipy.io.loadmat(os.path.join(self.root, self.base_folder, 'train_list.mat'))['annotation_list']
            targets = scipy.io.loadmat(os.path.join(self.root, self.base_folder, 'train_list.mat'))['labels']
        else:
            data = scipy.io.loadmat(os.path.join(self.root, self.base_folder, 'test_list.mat'))['annotation_list']
            targets = scipy.io.loadmat(os.path.join(self.root, self.base_folder, 'test_list.mat'))['labels']

        self.data = [item[0][0] for item in data]
        self.targets = [item[0]-1 for item in targets]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = os.path.join(self.root, self.base_folder, 'Images', img + '.jpg')

        img = Image.open(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.file_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        if not os.path.exists(os.path.join(root, self.base_folder, 'Images')):
            return False
        if not os.path.exists(os.path.join(root, self.base_folder, 'Annotation')):
            return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url+self.filenames[0], self.root, filename=self.filenames[0], md5=self.tgz_md5s[0])
        download_and_extract_archive(self.url+self.filenames[1], self.root, filename=self.filenames[1], md5=self.tgz_md5s[1])
        download_and_extract_archive(self.url+self.filenames[2], self.root, filename=self.filenames[2], md5=self.tgz_md5s[2])

class CropDogs(VisionDataset):
    """`Stanford Dataset <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``stanford-dogs`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'stanford-dogs'
    url = "http://vision.stanford.edu/aditya86/ImageNetDogs/"
    filenames = ["images.tar", "annotation.tar", "lists.tar"]
    tgz_md5s = ['1bb1f2a596ae7057f99d7d75860002ef', '4298cc0031f6bc6e74612ac83b5988e2', 'edbb9f16854ec66506b5f09b583e0656']
    file_list = [
        ['file_list.mat', 'ce0676c9520e4b7e1d43221cd9c647b0'],
        ['train_list.mat','d37f459eacccfa4d299373dffba9648d'],
        ['test_list.mat', '66f60c285efbc3ce2fb7893bd26c6b80'],
    ]

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):

        super(CropDogs, self).__init__(root)
        self.transform = transform
        self.target_transform = target_transform

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            data = scipy.io.loadmat(os.path.join(self.root, self.base_folder, 'train_list.mat'))['annotation_list']
            targets = scipy.io.loadmat(os.path.join(self.root, self.base_folder, 'train_list.mat'))['labels']
        else:
            data = scipy.io.loadmat(os.path.join(self.root, self.base_folder, 'test_list.mat'))['annotation_list']
            targets = scipy.io.loadmat(os.path.join(self.root, self.base_folder, 'test_list.mat'))['labels']

        data = [item[0][0] for item in data]
        targets = [item[0]-1 for item in targets]

        self.data, self.boxes, self.targets = [], [], []
        for idx, item in enumerate(data):
            xml_path = os.path.join(self.root, self.base_folder, 'Annotation', item)
            e = xml.etree.ElementTree.parse(xml_path).getroot()
            for objs in e.iter('object'):
                box = [int(objs.find('bndbox').find('xmin').text), int(objs.find('bndbox').find('ymin').text), int(objs.find('bndbox').find('xmax').text), int(objs.find('bndbox').find('ymax').text)]
                self.data.append(item)
                self.boxes.append(box)
                self.targets.append(targets[idx])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, box, target = self.data[index], self.boxes[index], self.targets[index]
        img = os.path.join(self.root, self.base_folder, 'Images', img + '.jpg')

        img = Image.open(img).convert('RGB')
        img = img.crop(box)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.file_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        if not os.path.exists(os.path.join(root, self.base_folder, 'Images')):
            return False
        if not os.path.exists(os.path.join(root, self.base_folder, 'Annotation')):
            return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url+self.filenames[0], self.root, filename=self.filenames[0], md5=self.tgz_md5s[0])
        download_and_extract_archive(self.url+self.filenames[1], self.root, filename=self.filenames[1], md5=self.tgz_md5s[1])
        download_and_extract_archive(self.url+self.filenames[2], self.root, filename=self.filenames[2], md5=self.tgz_md5s[2])

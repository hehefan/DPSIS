# Unsupervised Visual Representation Learning via Dual-level Progressive Similar Instance Selection (DPSIS)
![](https://github.com/hehefan/DPSIS/blob/main/imgs/framework.png)


## Data
data/cifar10: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

data/cifar100: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

data/svhn: http://ufldl.stanford.edu/housenumbers/train_32x32.mat, 
           http://ufldl.stanford.edu/housenumbers/test_32x32.mat
           
data/cub200: http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz

data/dogs: http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar, 
           http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar,
           http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar
           
data/ILSVRC2012: http://image-net.org/download-images

data/Places365: http://places2.csail.mit.edu/download.html

## Usage
### CIFAR10
```
python cifar.py -b 128--threshold-1=0.9 --threshold-2=0.6
```

### ImageNet
```
python imagenet.py data/ILSVRC2012 -b 256 --threshold-1=0.9 --threshold-2=0.6
```


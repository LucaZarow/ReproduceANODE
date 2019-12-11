import torch
import numpy as np
from mlxtend.data import loadlocal_mnist
from torch.utils.data import Dataset
import pickle as pickle
import os

# end2end kfold integration - passed to loader
class kFold():
    def __init__(self, images, labels, numFolds=5):
        self.data = self._pack(images, labels)
        self.numFolds = numFolds
    
    def _unpack(self, subset):
        data = []
        labels = []

        for x,y in subset:
            data.append(x)
            labels.append(y)

        data = np.array(data)
        labels = np.array(labels)
    
        return (data, labels)

    def _pack(self, data, labels):
        values = []
        
        for idx in range(len(data)):
            item = (data[idx], labels[idx])
            values.append(item)
        
        return np.array(values)
    
    
    def split(self):
        np.random.shuffle(self.data)
        
        splits = []
        folds = []
        splitPoint = self.data.shape[0] // (self.numFolds)
        
        for i in range(self.numFolds - 1):
            folds.append(self.data[i*splitPoint:(i+1)*splitPoint, :])
            
        folds.append(self.data[(i+1)*splitPoint:,:])
        
        foldDivisor = len(folds[0]) // 2
        for i in range(self.numFolds):
            train = []
            for k in range(self.numFolds):
                if i == k:
                    validation = folds[i][:foldDivisor] 
                    test = folds[i][foldDivisor:] 
                else:
                    train.append(folds[k])
            
            train = np.vstack(train)
            splits.append((
                self._unpack(train), 
                self._unpack(validation),
                self._unpack(test)
            ))
        
        return splits    

# torch dataset object
class torchSet(Dataset):
    def __init__(self, info):
        self.data, self.labels = info

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        return image, label

# MNIST auto-split generator                      
class MNIST():
    def __init__(self, img_path, label_path, num_folds):
        data, labels = loadlocal_mnist(images_path=img_path, labels_path=label_path)
        data = data.reshape(data.shape[0], 1, 28, 28).astype(np.float32)
        self.splits = kFold(data, labels, num_folds).split()

# CIFAR auto-split generator                     
class CIFAR():
    def __init__(self, data_path, num_folds):
        data, labels = self.load_CIFAR_set(data_path)
        self.splits = kFold(data, labels, num_folds).split()

    def _load_CIFAR_batch(self, filename):
        with open(filename, 'rb') as f:
            datadict = pickle.load(f, encoding='latin1')
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000, 3, 32, 32).astype(np.float32)
            Y = np.array(Y)
            return X, Y
        
    def load_CIFAR_set(self, rootdir):
        imgs = []
        labels = []
        for idx in range(1,6):
            fp = os.path.join(rootdir, 'data_batch_%d' % (idx, ))
            X, Y = self._load_CIFAR_batch(fp)
            imgs.append(X)
            labels.append(Y)
        imgs = np.concatenate(imgs)
        labels = np.concatenate(labels)
        return imgs, labels
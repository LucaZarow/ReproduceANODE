import torch
from mlxtend.data import loadlocal_mnist
from torch.utils.data import Dataset

#TODO: unit test file

# end2end kfold integration
# splits double packed - ((train_data, train_labels), (val_data, val_labels), (test_data, test_labels))
class kFold():
    def __init__(self, images, labels, numFolds=5):
        self.data = self.__pack__(images, data)
        self.numFolds = numFolds
    
    def __unpack__(subset):
        data = []
        labels = []

        for x,y in subset:
            data.append(x)
            labels.append(y)

        data = np.array(data)
        labels = np.array(labels)
    
        return (data, labels)

    def __pack__(data, labels):
        values = []
        
        for idx in range(len(data)):
            item = (data[idx], labels[idx])
            values.append(item)
        
        return np.array(values)
    
    
    def generateSplits(self):
        np.random.shuffle(self.data)
        
        splits = []
        folds = []
        splitPoint = self.data.shape[0] // (self.numFolds)
        
        for i in range(self.numFolds - 1):
            folds.append(self.data[i*splitPoint:(i+1)*splitPoint, :])
            
        folds.append(self.data[(i+1)*splitPoint:,:]
        
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
                self.__unpack__(train), 
                self.__unpack__(validation),
                self.__unpack__(test)
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
        data = data.reshape(data.shape[0], 28, 28)
        self.splits = kFold(data, labels, num_folds).generateSplits()

# TODO: LOAD CIFAR, verify reshape
# CIFAR auto-split generator                     
class CIFAR():
    def __init__(self, img_path, label_path, num_folds):
        data, labels = loadlocal_mnist(images_path=img_path, labels_path=label_path)
        data = data.reshape(data.shape[0], 32, 32)
        self.splits = kFold(data, labels, num_folds).generateSplits()
                         

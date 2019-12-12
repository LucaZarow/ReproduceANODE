import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from loader import torchSet
from models import *


#TODO: early stoppping? reduce on plateau? 
#Make model file with optim and loss inits 


class Trainer():
    def __init__(self, model, optimizer, data, device):
        self.model= model
        self.optimizer= optimizer
        self.data = data
        self.device = device
        
        self.train_metrics = {
            'legend' : "Train",
            'epochs' : None, 
            'loss' : {},
            'accuracy' : {}
        }
        self.val_metrics = {
            'legend' : "Validation",
            'epochs' : None,
            'loss' : {},
            'accuracy' : {}
        }
        self.test_metrics = None
            
    def metrics(self):
        return [self.train_metrics, self.val_metrics]
    
    def _initModel(self, params):
        return self.model(*params)
    
    def _initOptimizer(self, model, lr):
        return self.optimizer(model.parameters(), lr)
    
    def train(self, model_params, learning_rate, epochs, batch_size, num_workers, verbose=True, checkpoint=True, num_loss = 5):
        best_vals = [0,0,0,0,0] #static for 5fold - to be adapted
        for idx, split in enumerate(self.data.splits):           
            train, validation, _ = split
            train_data = torchSet(train)
            val_data = torchSet(validation)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, \
                                                       num_workers=num_workers, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, \
                                                     num_workers=num_workers)
           
            torch.cuda.empty_cache()
            model = self._initModel(model_params).to(self.device)
            optimizer = self._initOptimizer(model, learning_rate)
            loss = nn.CrossEntropyLoss().to(self.device)

            best_score = 0
            train_losses = []
            train_accuracies = []
            val_losses = []
            val_accuracies = []
            self.train_metrics['epochs'] = epochs
            self.val_metrics['epochs'] = epochs
            
            for epoch in range(epochs):
                
                model.train()
                correct = 0.
                total = 0.
                running_loss = 0.0
                avg_loss = 0.0

                for i, (item, target) in enumerate(train_loader):
            
                    item = torch.Tensor(item)
                    item = item.to(device=self.device, dtype=torch.float32)
                    target = target.to(device=self.device, dtype=torch.int64)


                    optimizer.zero_grad()
                    output = model(item)
                    error = loss(output, target)
                    error.backward()
                    optimizer.step()
                    
                    running_loss += error.item()
                    avg_loss += error.item()
                    
                    printer = len(train_data.data) // (num_loss * batch_size)
                    if (i % printer == printer - 1):
                        if(verbose):
                            print('[%d, %5d] loss: %.5f' %
                                  (epoch + 1, i + 1, running_loss / printer))
                        running_loss = 0.0

                    preds = F.softmax(output, dim=1)
                    preds_cls = preds.argmax(dim=1)
                    correct_preds = torch.eq(preds_cls, target)
                    correct += torch.sum(correct_preds).detach().cpu().item()
                    total += len(correct_preds)
                    
                
                train_acc = correct / total
                train_accuracies.append(train_acc)
                train_losses.append(avg_loss / total)
                print("[Fold "+str(idx+1)+"] Epoch:"+str(epoch+1)+" Training Acc:"+str(train_acc))

                model.eval()
                correct = 0.0
                total = 0.0
                avg_loss = 0.0

                for i, (item, target) in enumerate(val_loader):
           
                    item = torch.Tensor(item)
                    item = item.to(device=self.device, dtype=torch.float32)
                    target = target.to(device=self.device, dtype=torch.int64)
           
                    output = model(item)
                    error = loss(output, target)
                    avg_loss += error.item()
                    
                    preds = F.softmax(output, dim=1)
                    preds_cls = preds.argmax(dim=1)
                    correct_preds = torch.eq(preds_cls, target)
                    correct += torch.sum(correct_preds).detach().cpu().item()
                    total += len(correct_preds)

                valid_acc = correct / total
                val_accuracies.append(valid_acc)
                val_losses.append(avg_loss / total)
                print("[Fold "+str(idx+1)+"] Epoch:"+str(epoch+1)+" Validation Acc:"+str(valid_acc))
        
            if(valid_acc > best_score and checkpoint):
                best_score = valid_acc
                best_vals[idx] = valid_acc
                torch.save(model.state_dict(), './models/fold_'+str(idx+1)+'.pth.tar')
            
            self.train_metrics['loss']['fold'+str(idx+1)] = train_losses
            self.train_metrics['accuracy']['fold'+str(idx+1)] = train_accuracies
            self.val_metrics['loss']['fold'+str(idx+1)] = val_losses
            self.val_metrics['accuracy']['fold'+str(idx+1)] = val_accuracies
            
        torch.cuda.empty_cache()
        print("Best Fold Validation Results:", np.round_(best_vals, 5))
        print("Finished Cross Validation Training")
    
    def test(self, model_params, batch_size, num_workers):
        test_results = []
        for idx, split in enumerate(self.data.splits): 
            
            _, _, test = split
            test_data = torchSet(test)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, \
                                                       num_workers=num_workers)
            torch.cuda.empty_cache()
            model = self._initModel(model_params)
            model.load_state_dict(torch.load("./models/fold_"+str(idx+1)+'.pth.tar'))
            model = model.to(self.device)
            
            model.eval()
            correct = 0.
            total = 0.
            
            for i, (item, target) in enumerate(test_loader):

                item = torch.Tensor(item)
                item = item.to(device=self.device, dtype=torch.float32)
                target = target.to(device=self.device, dtype=torch.int64)

                output = model(item)
                preds = F.softmax(output, dim=1)
                preds_cls = preds.argmax(dim=1)
                correct_preds = torch.eq(preds_cls, target)
                correct += torch.sum(correct_preds).detach().cpu().item()
                total += len(correct_preds)
                
            test_acc = correct / total
            test_results.append(test_acc)
            print("[Fold: "+str(idx+1)+"] Testing Acc:", test_acc)
        self.test_metrics = test_results
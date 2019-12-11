import torch
import torch.nn.functional as F
from loader import torchSet
from models import *


#TODO: early stoppping? reduce on plateau? 
#Make model file with optim and loss inits 


class Trainer():
    def __init__(self, data, device):
        self.data = data
        self.device = device

    def train(self, epochs, batch_size, num_workers, verbose=True, checkpoint=True):
        best_vals = [0,0,0,0,0]
        for idx, split in enumerate(self.data.splits): 
            
            train, validation, _ = split
            train_data = torchSet(train)
            val_data = torchSet(validation)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, /
                                                       num_workers=self.num_workers, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size, 
                                                     num_workers=self.num_workers)
            
            torch.cuda.empty_cache()
            model = models.ANODE().to(self.device)
            loss = models.loss().to(self.device)
            optimizer = models.optimize()
            
            best_score = 0
            for epoch in range(epochs):
                
                net.train()
                correct = 0.
                total = 0.
                running_loss = 0.0

                for i, (item, target) in enumerate(train_loader):
            
                    item = torch.Tensor(digits)
                    item = item.to(device=self.device, dtype=torch.float32).unsqueeze(1)
                    target = target.to(device=self.device, dtype=torch.int64)


                    optimizer.zero_grad()
                    output = model(item)
                    error = loss(output, target)
                    error.backward()
                    optimizer.step()
                    
                    running_loss += error.item()
                    printer = len(train_data.data) // 5
                    if (i % printer == printer - 1):
                        if(verbose):
                            print('[%d, %5d] loss: %.5f' %
                                  (epoch + 1, i + 1, running_loss / 50))
                        running_loss = 0.0

                    preds = F.softmax(output, dim=1)
                    preds_cls = preds.argmax(dim=1)
                    correct_preds = torch.eq(preds_cls, target)
                    correct += torch.sum(correct_preds).detach().cpu().item()
                    total += len(correct_preds)
                    
                train_acc = correct / total
                print("[Fold "+str(idx+1)+"] Epoch:"+str(epoch+1)+" Training Acc:"+str(train_acc))

                net.eval()
                correct = 0.
                total = 0.

                for i, (item, target) in enumerate(val_loader):
           
                    item = torch.Tensor(item)
                    item = digits.to(device=self.device, dtype=torch.float32).unsqueeze(1)
                    target = target.to(device=self.device, dtype=torch.int64)
           
                    output = net(item)
                    preds = F.softmax(output, dim=1)
                    preds_cls = preds.argmax(dim=1)
                    correct_preds = torch.eq(preds_cls, target)
                    correct += torch.sum(correct_preds).detach().cpu().item()
                    total += len(correct_preds)

                valid_acc = correct / total
                print("[Fold "+str(idx+1)+"] Epoch:"+str(epoch+1)+" Training Acc:"+str(valid_acc))
        
            if(valid_acc > best_score and checkpoint):
                best_score = valid_acc
                best_vals[s] = valid_acc
                torch.save(net.state_dict(), './models/fold_'+str(idx+1)+'.pth.tar')
                early_stop_criterion = 0
        
        torch.cuda.empty_cache()
        print("Fold Validation Results:", best_vals)
        print("Finished Cross Validation Training")
    
    def test(self, batch_size, num_workers):
        for idx, split in enumerate(self.data.splits): 
            
            _, _, test = split
            test_data = torchSet(test)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, /
                                                       num_workers=self.num_workers)
            
            torch.cuda.empty_cache()
            model = models.ANODE()
            model.load_state_dict(torch.load("./models/fold_"+str(idx+1)+'.pth.tar'))
            model = model.to(self.device)
            
            for i, (item, target) in enumerate(test_loader):

                item = torch.Tensor(item)
                item = item.to(device=self.device, dtype=torch.float32).unsqueeze(1)
                target = target.to(device=self.device, dtype=torch.int64)

                output = model(item)
                preds = F.softmax(output, dim=1)
                preds_cls = preds.argmax(dim=1)
                correct_preds = torch.eq(preds_cls, target)
                correct += torch.sum(correct_preds).detach().cpu().item()
                total += len(correct_preds)
                
            test_acc = correct / total
            print("[Fold: "+str(idx+1)+"] Testing Acc:", test_acc)
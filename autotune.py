import training
import numpy as np

class Autotuner():
    def __init__(self, model, optimizer, data, device):
        self.trainer = training.Trainer(model, optimizer, data, device)
        self.logs = []
        
    def _tune(self, model_params, training_params):
        learning_rate, epochs, batch_size, num_workers = training_params
        self.trainer.train(model_params, learning_rate, epochs, batch_size, num_workers, verbose=False)
        self.trainer.test(model_params, batch_size, num_workers)
        results = np.mean(self.trainer.test_metrics)
        log = {
            'model' : model_params,
            'train' : training_params,
            'results' : results
        }
        self.logs.append(log)
    
    def search(self, configs):
        for idx, config in enumerate(configs):
            model_params = config['model']
            training_params = config['train']
            self._tune(model_params, training_params)
            
    def bestConfig(self):
        best_id = -1
        best_score = -1
        for idx, log in enumerate(self.logs):
            if log['results'] > best_score:
                best_score = log['results']
                best_id = idx
        return self.logs[best_id]
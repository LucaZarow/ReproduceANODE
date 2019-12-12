import numpy as np
import matplotlib.pyplot as plt

class Plotter():
    def __init__(self, metrics=None):
        self.losses = []
        self.accuracies = []
        self.epochs = None
        
        if metrics is not None:
            self._setMetrics(metrics)
        
    def _setMetrics(self, metrics):
        if type(metric) is not list:
            metrics = list(metrics)
        
        self.epochs = metrics[0]['epochs']
        
        for metric in metrics
            folded_losses = []
            folded_accuracies = []
            for key in metric['loss'].keys():
                folded_losses.append(metrics['loss'][key])
                folded_accuracies.append(metrics['accuracies'][key])
            self.losses.append(folded_losses)
            self.accuracies.append(folded_accuracies)
        
    def _plot(self, values, title, xlabel, ylabel):
        if type(values) is not list:
            values = list(values)
        
        for value in values:
            mu = np.mean(value, axis=1)
            sig = np.std(value, axis=1)
            plt.plot(self.epochs, mean)
            plt.fill_between(self.epochs, mu+sig, mu-sigma, alpha=0.5)
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

        
    def plotLoss(title, metrics=None):
        if metrics is not None:
            self._setMetrics(metrics)
        self._plot(self.losses, title)
        
    def plotAccuracy(title, metrics=None):
        if metrics is not None:
            self._setMetrics(metrics)
        self._plot(self.accuracies, title)
            
       
        
        
        
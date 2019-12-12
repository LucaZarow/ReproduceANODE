import numpy as np
import matplotlib.pyplot as plt

class Plotter():
    def __init__(self, metrics=None):
        self.losses = None
        self.accuracies = None
        self.epochs = None
        self.legend = None
        
        if metrics is not None:
            self._setMetrics(metrics)
        
    def _setMetrics(self, metrics):
        self.losses = []
        self.accuracies = []
        self.legend = []
        
        if type(metrics) is not list:
            metrics = [metrics]
        
        self.epochs = metrics[0]['epochs']
        for metric in metrics:
            folded_losses = []
            folded_accuracies = []
            self.legend.append(metric['legend'])
            for key in metric['loss'].keys():
                folded_losses.append(metric['loss'][key])
                folded_accuracies.append(metric['accuracy'][key])
            self.losses.append(folded_losses)
            self.accuracies.append(folded_accuracies)
        
    def _plot(self, values, title, xlabel, ylabel, fig_name=None):
        if type(values) is not list:
            values = [values]
        
        plt.figure(figsize=(5,5))
        
        for value in values:
            mu = np.mean(value, axis=0)
            sig = np.std(value, axis=0)
            plt.plot(np.arange(1,self.epochs+1), mu)
            plt.fill_between(np.arange(1,self.epochs+1), mu+sig, mu-sig, alpha=0.5)
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(self.legend)
        if fig_name is not None:
            plt.figsave(title+'.png')
        plt.show()
        plt.close()
        
    def plotLoss(self, title, metrics=None, fig_name=None):
        xlabel = "Epochs"
        ylabel = "Loss"
        if metrics is not None:
            self._setMetrics(metrics)
        self._plot(self.losses, title, xlabel, ylabel, fig_name)
        
    def plotAccuracy(self, title, metrics=None, fig_name=None):
        xlabel = "Epochs"
        ylabel = "Accuracy"
        if metrics is not None:
            self._setMetrics(metrics)
        self._plot(self.accuracies, title, xlabel, ylabel, fig_name)
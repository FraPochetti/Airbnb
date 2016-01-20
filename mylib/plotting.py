import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_matrix(predictions, truth,  le):
    cm = confusion_matrix(truth, predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm_normalized, interpolation='nearest',cmap='viridis')
    plt.colorbar()
    rng = np.unique(truth)
    plt.xticks(range(len(rng)), le.inverse_transform(rng))
    plt.yticks(range(len(rng)), le.inverse_transform(rng))

    plt.show()

def precision_score(pred, truth,  le):
    cm = confusion_matrix(truth, pred)
    rng = np.unique(truth)
    for i, lab in enumerate(le.inverse_transform(rng)):
        print lab, float(cm[i, i])/sum(cm[i, :]), 'true positives', cm[i, i], 'total positives', sum(cm[i, :])

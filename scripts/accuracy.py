from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(is_dnn,y_true, y_pred, classes,normalize=True,title=None,cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    cm = confusion_matrix(y_true, y_pred)
    if(is_dnn):
      y_pred = y_pred.argmax(axis=1)
      y_true = y_true.argmax(axis=1)
      classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, ax

def get_accuracy(classifier,x_test, y_test):
  loss, accuracy = classifier.evaluate(x_test, y_test)
  print('Accuracy: %f, Loss: %f' % (accuracy*100,loss*100))


def get_metrics(test_labels, predicted_labels):
    
    print('Accuracy:', np.round(
                        metrics.accuracy_score(test_labels, 
                                               predicted_labels),
                        4))
    print('Precision:', np.round(
                        metrics.precision_score(test_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        4))
    print('Recall:', np.round(
                        metrics.recall_score(test_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        4))
    print('F1 Score:', np.round(
                        metrics.f1_score(test_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        4))
                            
def display_classification_report(test_labels, predicted_labels, classes):

    report = metrics.classification_report(y_true=test_labels, y_pred=predicted_labels) 
    print(report)
    
      
def display_model_performance_metrics(classifier,x_data,test_labels, classes,is_dnn,name,save):
  predicted_labels = classifier.predict(x_data)
  if(is_dnn):
    get_accuracy(classifier,x_data,test_labels)
  else:
    print('Model Performance metrics:')
    print('-'*30)
    get_metrics(test_labels, predicted_labels)
    print('\nModel Classification report:')
    print('-'*30)
    display_classification_report(test_labels, predicted_labels,classes)
  fig, ax = plot_confusion_matrix(is_dnn,test_labels, predicted_labels, classes, normalize=True)
  fig.set_size_inches(18.5, 10.5)
  if(save):
    fig.savefig(f'/content/drive/My Drive/Emotion-Detection/{name}.png')
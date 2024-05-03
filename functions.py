import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score, accuracy_score,confusion_matrix, classification_report
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt

def read_serialized_file(file_path):
  directory = file_path 
  infile = open(directory,'rb')
  loaded_data = pickle.load(infile)
  infile.close()
  return loaded_data

def get_all_accuracies(model, X_test, y_test):
    test_values = [np.count_nonzero(y_test == i) for i in np.unique(y_test)]
    min_value = min(test_values)
    test_weights = [min_value / value for value in test_values]
    test_class_weights = {label: weight for label, weight in enumerate(test_weights)}
    total_weights = sum(test_class_weights.values())
    
    classwise_accuracy = {}
    b = model.predict(X_test)
    y_pred_labels = np.argmax(b, axis=1)
    
    for class_label in np.unique(y_test):
        class_indices = np.where(y_test == class_label)[0]
        class_accuracy = accuracy_score(y_test[class_indices], y_pred_labels[class_indices])
        classwise_accuracy[class_label] = class_accuracy
    
    class_wise_acc_array = []
    total_acc = 0
    for class_label, accuracy in classwise_accuracy.items():
        print(f"Class {class_label}: Accuracy = {accuracy}")
        class_wise_acc_array.append(accuracy)
        total_acc += accuracy*test_class_weights[class_label]

    average_accuracy = total_acc/total_weights
    print("Weighted Accuracy = ", average_accuracy)
    
    return [average_accuracy] + class_wise_acc_array 

def get_all_metrics(model, X_test, y_test):
    X_test = np.array(X_test)
    actual = y_test
    y_test = np.array(y_test)
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    pred = model.predict(X_test)
    predicted = np.argmax(pred, axis=1)
    
    labels = [0, 1, 2, 3,4, 5, 6, 7, 8]
    
    f1 = f1_score(actual, predicted,labels=labels, average=None)
    precision = precision_score(actual, predicted,labels=labels, average=None)
    recall = recall_score(actual, predicted,labels=labels,average=None)
    
    average_f1 = f1_score(actual, predicted, average="weighted")
    average_precision = precision_score(actual, predicted, average="weighted")
    average_recall = recall_score(actual, predicted, average="weighted")
    
    print('Classwise metrics')
    print("f1 \n",f1)
    print("precision \n",precision)
    print("recall \n",recall)
    
    print('\nOverall metrics')
    print("f1 \n",average_f1)
    print("precision \n",average_precision)
    print("recall \n",average_recall)
    
    return [average_precision,average_recall,average_f1] + precision.tolist() + recall.tolist() + f1.tolist()

def plot_metrics(history, max_epochs=80):  # Add max_epochs as a parameter
    
  metrics = ['loss', 'accuracy']
  mpl.rcParams['figure.figsize'] = (12, 10)
  colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch[:max_epochs], history.history[metric][:max_epochs], color=colors[0], label='Train')  # Limit the data to the first 80 epochs
    plt.plot(history.epoch[:max_epochs], history.history['val_'+metric][:max_epochs],
             color='orange', label='Val')  # Limit the data to the first 80 epochs
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      ylim_max = 1.05  # Adjust the padding value as needed
      plt.ylim([0, ylim_max])
      y_ticks = np.arange(0, ylim_max, 0.2)  # Define y-axis ticks with 0.1 increments
      plt.yticks(y_ticks)
    plt.grid(linestyle='--', linewidth=0.5, color='gray')
    plt.legend()
    plt.savefig('metrics.png')
    
def get_classification_result(model, X_test, y_test):
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    overall_accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_0_accuracy = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    class_1_accuracy = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
    
    print(f"Class 0 Accuracy: {class_0_accuracy:.4f}")
    print(f"Class 1 Accuracy: {class_1_accuracy:.4f}")
    print()
    
    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)
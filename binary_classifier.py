import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D, Dense, Dropout, Flatten,  BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from keras import callbacks
import numpy as np
import pandas as pd
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from functions import plot_metrics, read_serialized_file,get_classification_result
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


stacked_spects = read_serialized_file("./Pickle Files/stacked_specs_2026_norm.pkl")
feature_data = pd.DataFrame(stacked_spects,columns=['feature','class','file'])

X = np.array(feature_data['feature'].tolist())
y = np.array(feature_data['class'].tolist())
f = np.array(feature_data['file'].tolist())

X_train, X_, y_train, y_ = train_test_split(X,y,test_size=0.2,random_state=42, stratify=y)

X_test, X_validate, y_test, y_validate = train_test_split(X_,y_,test_size=0.5,random_state=42, stratify=y_)

print(np.shape(X_train))
print(np.shape(X_test))
print(np.shape(X_validate))

train_values = [np.count_nonzero(y_train == i) for i in np.unique(y_train)]
print(train_values)

disease_stats = [780, 840]
min_value = min(disease_stats)

weights = [min_value / value for value in disease_stats]
class_weights = {label: weight for label, weight in enumerate(weights)}
sample_weight=np.array([class_weights[label] for label in y_train])

def create_model(input_shape = (128, 264, 3)):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(128, 264, 3)))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3),padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(64, (3, 3),padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3),padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# model = create_model()
# earlystopping = callbacks.EarlyStopping(monitor ="val_loss", mode ="min", patience = 8, restore_best_weights = True, verbose=1)
# baseline_history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_validate, y_validate), callbacks =[earlystopping], sample_weight=sample_weight)
# get_classification_result(model, X_test, y_test)
# plot_metrics(baseline_history,100)

# model.save('./Pickle Files/binary_model_v2.h5')
# model_loaded = tf.keras.models.load_model('./Pickle Files/binary_model_v1.h5')
model_loaded = tf.keras.models.load_model('./Pickle Files/binary_model_v2.h5')

# get_classification_result(model_loaded, X_test, y_test)

# preds_single=[]
# for i in range(len(y_test)):
#     single_sample = np.array([X_test[i]])
#     y_pred_prob = model_loaded.predict(single_sample, batch_size = 1)
#     y_pred_single = (y_pred_prob > 0.5).astype(int)
#     preds_single.append(y_pred_single[0].tolist())
    
# print("Individual prediction accuracy",accuracy_score(y_test, preds_single))

def plot_confusion_matrix(loded_model, X_test, y_test):
    pred = loded_model.predict(X_test)
    y_pred_binary = (pred >= 0.5).astype(int)

    conf_matrix = confusion_matrix(y_test, y_pred_binary)

    plt.figure(figsize=(5,4))
    fx=sns.heatmap(conf_matrix,fmt="d", annot=True,cmap="Blues", cbar=False, 
                xticklabels=["Abnormal", "Healthy"], yticklabels=["Abormal", "Healthy"],annot_kws={"size": 16})

    fx.set_title('Confusion Matrix \n', fontsize=15);
    fx.set_xlabel('\n Predicted Values\n', fontsize=15)
    fx.set_ylabel('Actual Values\n', fontsize=15);
    plt.show()
    plt.savefig('confusion_matrix7.png')
    
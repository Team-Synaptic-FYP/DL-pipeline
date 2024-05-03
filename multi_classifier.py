import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D, Dense, Dropout, Flatten,  BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from keras import callbacks

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from functions import plot_metrics, read_serialized_file, get_all_accuracies, get_all_metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import pickle


stacked_spects = read_serialized_file("./Pickle Files/stacked_specs_975_norm.pkl")
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

disease_stats = [89, 84, 131, 85, 92, 91, 83, 128, 81]
min_value = min(disease_stats)
weights = [min_value / value for value in disease_stats]
class_weights = {label: weight for label, weight in enumerate(weights)}
sample_weight=np.array([class_weights[label] for label in y_train])

def create_model(input_shape = (128, 264, 3)):
    model_holdout = Sequential()

    model_holdout.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model_holdout.add(BatchNormalization())
    model_holdout.add(MaxPooling2D((2, 2)))
    
    model_holdout.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
#     model_holdout.add(BatchNormalization())
    model_holdout.add(MaxPooling2D((2, 2)))

    model_holdout.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#     model_holdout.add(BatchNormalization())
    model_holdout.add(MaxPooling2D((2, 2)))
    
    model_holdout.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#     model_holdout.add(BatchNormalization())
    model_holdout.add(MaxPooling2D((2, 2)))

    model_holdout.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#     model_holdout.add(BatchNormalization())
    model_holdout.add(MaxPooling2D((2, 2)))

    model_holdout.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#     model_holdout.add(BatchNormalization())
    model_holdout.add(MaxPooling2D((2, 2)))
    
    model_holdout.add(Conv2D(128, (3, 3), padding='same', activation='relu'))

    model_holdout.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    
#     model_holdout.add(Dropout(0.25))

    model_holdout.add(GlobalAveragePooling2D())

    model_holdout.add(Dense(9, activation='softmax'))
    model_holdout.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model_holdout


# model_holdout = create_model()
# earlystopping = callbacks.EarlyStopping(monitor ="val_loss", mode ="min", patience = 10, restore_best_weights = True, verbose=1)
# baseline_history = model_holdout.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_validate, y_validate), callbacks =[earlystopping], sample_weight=sample_weight)

# model_holdout.save('./Pickle Files/disease_model_v1.h5')
# model_path = './Pickle Files/disease_model_v1.joblib'
# joblib.dump(model_holdout, model_path)

# pkl_model_path = './Pickle Files/disease_model_v1.pkl'

# with open(pkl_model_path, "wb") as file:
#     pickle.dump(pkl_model_path, file)

model = tf.keras.models.load_model('./Pickle Files/disease_model_v1.h5')
predictions = model.predict(X_test)

# Example: Getting the predicted class labels
predicted_labels = np.argmax(predictions, axis=1)

# Compute the accuracy score
accuracy = accuracy_score(y_test, predicted_labels)

print("Accuracy:", accuracy)

preds_single=[]
for i in range(len(y_test)):
    single_sample = np.array([X_test[i]])
    y_pred_prob = model.predict(single_sample, batch_size = 1)
    pred = np.argmax(y_pred_prob, axis=1)
    preds_single.append(pred[0])
    
print(preds_single)
print("Accuracy:", accuracy_score(y_test, preds_single))
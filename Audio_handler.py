import tensorflow as tf
import pandas as pd
import librosa
import pickle
import numpy as np

final_csv_file_path = './MetaData_975.csv'
final_csv_file_path_binary = './MetaData_Binary_2026.csv'


def load_audio_files(file_name):
  labeled_audio_array = []
  df = pd.read_csv('./' + file_name)
  # audio_file_path = './Audio/' #for disease categorization model training
  audio_file_path = './New Audio Binary/' #for dbinary classification model training

  for index, row in df.iterrows():
    fileName = row['FileName']
    _audio_path = audio_file_path + str(fileName) + ".wav"
    _label = row['Label']

    y, sr = librosa.load(_audio_path, sr=22500, duration=6)

    #Z score normalization
    mean = np.mean(y)
    std = np.std(y)
    norm_audio = (y - mean) / std

    labeled_audio_array.append([norm_audio,_label,fileName])

  return labeled_audio_array

def create_multi_audio_array ():
  labeled_audio_array = load_audio_files("MetaData_975.csv")

  labeled_audio = './Pickle Files/label_enc_audio_975_norm.pkl'
  with open(labeled_audio, "wb") as file:
      pickle.dump(labeled_audio_array, file)

  print(f"Array serialized and saved to '{labeled_audio}'.")
  
 
def create_binary_audio_array ():
  labeled_audio_array = load_audio_files("MetaData_Binary_2026.csv")

  labeled_audio = './Pickle Files/label_enc_audio_binary_2026_norm.pkl'
  with open(labeled_audio, "wb") as file:
      pickle.dump(labeled_audio_array, file)

  print(f"Array serialized and saved to '{labeled_audio}'.")  

  # with open(labeled_audio, "rb") as file:
  #     loaded_array = pickle.load(file)
  # print(loaded_array)
  # print("Deserialized array:")
# create_multi_audio_array()
create_binary_audio_array()
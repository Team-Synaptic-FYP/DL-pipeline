import pickle
import librosa
import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score, accuracy_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

from functions import read_serialized_file

def generate_mel_spec(audio):
  mel = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128, n_fft=2048, hop_length=512))
  return mel

def generate_mfcc(audio):
  mfcc = librosa.feature.mfcc(y=audio, n_mfcc=128, n_fft=2048, hop_length=512)
  return mfcc

def generate_chroma(audio):
  chroma = librosa.feature.chroma_stft(y=audio, sr=22050, n_chroma=128, n_fft=2048, hop_length=512)
  return chroma

def create_augmented_features_imbalanced_audio():
  labelled_features = []

  audio_array = read_serialized_file("./Pickle Files/label_enc_audio_975_norm.pkl")
  for raw_audio, label, file in audio_array:
    melsp = generate_mel_spec(raw_audio)
    mfcc = generate_mfcc(raw_audio)
    chroma = generate_chroma(raw_audio)

    three_chanel = np.stack((melsp, mfcc, chroma), axis=2)

    labelled_features.append([three_chanel, label, file])
    
    if(label == 3 or label == 6):
        # Shift the pitch
        y_shifted = librosa.effects.pitch_shift(raw_audio, sr=22050, n_steps=1)
        melsp_shifted = generate_mel_spec(y_shifted)
        mfcc_shifted = generate_mfcc(y_shifted)
        chroma_shifted = generate_chroma(y_shifted)
        three_chanel_shifted = np.stack((melsp_shifted, mfcc_shifted, chroma_shifted), axis=2)
        labelled_features.append([three_chanel_shifted, label, file])

  print(len(labelled_features))

  return labelled_features

def create_features():
  labelled_features = []
  audio_array = read_serialized_file("./Pickle Files/label_enc_audio_binary_2026_norm.pkl")

  for raw_audio, label,file in audio_array:
    melsp = generate_mel_spec(raw_audio)
    mfcc = generate_mfcc(raw_audio)
    chroma = generate_chroma(raw_audio)

    three_chanel = np.stack((melsp, mfcc, chroma), axis=2)

    labelled_features.append([three_chanel, label, file])

  print(len(labelled_features))

  return labelled_features


def create_multi_clssifier_features():
  stacked_labeled_features = create_augmented_features_imbalanced_audio()

  stacked_features_path = './Pickle Files/stacked_specs_975_norm.pkl'
  with open(stacked_features_path, "wb") as file:
      pickle.dump(stacked_labeled_features, file)
      
def create_binary_clssifier_features():
  stacked_labeled_features = create_features()

  stacked_features_path = './Pickle Files/stacked_specs_2026_norm.pkl'
  with open(stacked_features_path, "wb") as file:
      pickle.dump(stacked_labeled_features, file)
      
create_binary_clssifier_features()
    
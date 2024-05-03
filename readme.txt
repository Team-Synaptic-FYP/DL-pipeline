The deep learning pipeline was developed in a conda enviornment.

Audio_handler - Used to load audio files using librosa and serialize
Spectrogram_handler - Used for feature extraction. All features were serialized for future use
binary_classifier - The CNN model used for disease detection
multi_classifier - The CNN model used for disease categorization in to 9 diseases
xai - Used for interpretation of model prediction

### Steps to run using a conda environment ###

    conda create --name <env_name> python=3.10
    conda activate env_name
    pip install -r requirements.txt

    #python <filename>.py

Without a virtual environment, just use pip install -r requirements.txt command and run the files required

import os
import pandas as pd
from zipfile import ZipFile
import joblib 
import numpy as np
import pydub
import joblib 
import librosa
import glob

file_name2 = "/content/drive/MyDrive/Dataset/Al-Quran App Data/AL Quran App (Voice Recognition)/5 Samples Data/flac_5_Sample.zip"

with ZipFile(file_name2, 'r') as zip:
  zip.extractall()
  print('Done')

#/content/.zip
audioFolder = glob.glob("/content/*.zip")
audioFolder

for x in audioFolder:
  with ZipFile(x, 'r') as zip:
    zip.extractall()
  
print('Done')

#/content/.mp3
audio_mp3 = glob.glob("/content/*.mp3")
audio_mp3

len(audio_mp3)

#Converting MP3 to flac format
from os import path
from pydub import AudioSegment

# files 
for f in audio_mp3:                                                                        
  src = f
  dst = str(f[:-3])+"flac"

# convert wav to mp3                                                            
  sound = AudioSegment.from_mp3(src)
  sound.export(dst, format="flac")

#/content/.flac
audio_flac = glob.glob("/content/*.flac")
audio_flac

cd /content/drive/MyDrive/Dataset/Al-Quran App Data/AL Quran App (Voice Recognition)/5 Samples Data/

# create a ZipFile object
#Stroing Data in a zip format at Drive
zipObj = ZipFile('flac_5_Sample.zip', 'w')
for k in audio_flac:

  # Add multiple files to the zip
  zipObj.write(k)
  
# close the Zip File
zipObj.close()

cd /content

filelist = audio_flac
#read them into pandas
Total_data = pd.DataFrame()
Total_data["file"] = filelist
Total_data.head()

audio_flac

temp = []
for var in audio_flac:
  temp.append(var.split("1")[0][9:-1])
  #print(var.split("1")[0][9:-1])
temp

Total_data["speaker"] = temp
Total_data.head(100)

cd /content/drive/MyDrive/Dataset/Al-Quran App Data/AL Quran App (Voice Recognition)/5 Samples Data/

Total_data.head()

#Encoding Data
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
encoding_label = le.fit_transform(Total_data['speaker'])
Total_data['speaker_Id'] = encoding_label
Total_data.head(10)

# Save the Encoding as a pickle in a file 
joblib.dump(le, 'Encoding_ID.pkl')

# Load the model from the file 
Total_data = joblib.load('/content/drive/MyDrive/Dataset/Al-Quran App Data/AL Quran App (Voice Recognition)/5 Samples Data/Dataframe_file_name.pkl') 
Total_data.head()

le = joblib.load("/content/drive/MyDrive/Dataset/Al-Quran App Data/AL Quran App (Voice Recognition)/5 Samples Data/Encoding_ID.pkl")



# Save the Dataframe as a pickle in a file 
joblib.dump(Total_data, 'Dataframe_file_name.pkl')

train, validate, test = \
              np.split(Total_data.sample(frac=1, random_state=42), 
                       [int(.7*len(Total_data)), int(.85*len(Total_data))])

train.index

def extract_features(files):
    
    # Sets the name to be the path to where the file is in my computer
    file_name = os.path.join(os.path.abspath(str ("/content/" + files.file)))
# Loads the audio file as a floating point time series and assigns the default sample rate
    # Sample rate is set to 22050 by default
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
# Generate Mel-frequency cepstral coefficients (MFCCs) from a time series 
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
# Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    stft = np.abs(librosa.stft(X))
# Computes a chromagram from a waveform or power spectrogram.
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
# Computes a mel-scaled spectrogram.
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
# Computes spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
# Computes the tonal centroid features (tonnetz)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    return mfccs, chroma, mel, contrast, tonnetz

#Traing on total dataset
Total_data_features = Total_data.apply(extract_features, axis=1)
#/content/content/abdallah-matroud-109-al-kafiroon-338-275.flac

cd /content/drive/MyDrive/Dataset/Al-Quran App Data/AL Quran App (Voice Recognition)/5 Samples Data

# Save the Total_data_features as a pickle in a file 
joblib.dump(Total_data_features, 'Total_data_features.pkl')

train_features = train.apply(extract_features, axis=1)

validate_features = validate.apply(extract_features, axis=1)

test_features = test.apply(extract_features, axis=1)

validate_features

#Total Dataset Training
features_Total_data = []
for i in Total_data.index:
    features_Total_data.append(np.concatenate((
        Total_data_features[i][0],
        Total_data_features[i][1], 
        Total_data_features[i][2], 
        Total_data_features[i][3],
        Total_data_features[i][4]), axis=0))

features_validate = []
for i in validate.index:
    features_validate.append(np.concatenate((
        validate_features[i][0],
        validate_features[i][1], 
        validate_features[i][2], 
        validate_features[i][3],
        validate_features[i][4]), axis=0))

features_train = []
for i in train.index:
    features_train.append(np.concatenate((
        train_features[i][0],
        train_features[i][1], 
        train_features[i][2], 
        train_features[i][3],
        train_features[i][4]), axis=0))

features_test = []
for i in test.index:
    features_test.append(np.concatenate((
        test_features[i][0],
        test_features[i][1], 
        test_features[i][2], 
        test_features[i][3],
        test_features[i][4]), axis=0))

X_Total_data = np.array(features_Total_data)

#X_Total_data = np.array(features_Total_data)
X_train = np.array(features_train)
X_test = np.array(features_test)
X_val = np.array(features_validate)
print(len(X_train))
print(len(X_test))
print(len(X_val))

y_Total_data= np.array(Total_data['speaker_Id'])

y_train = np.array(train['speaker_Id'])
y_val = np.array(validate['speaker_Id'])
y_test = np.array(test['speaker_Id'])

y_Total_data

#hot encode y
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
# Hot encoding y
lb = LabelEncoder()
#y_train = to_categorical(lb.fit_transform(y_train))
#y_val = to_categorical(lb.fit_transform(y_val))
y_Total_data =to_categorical(lb.fit_transform(y_Total_data))

len(y_Total_data)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
#X_train = ss.fit_transform(X_train)
#X_val = ss.transform(X_val)
#X_test = ss.transform(X_test)
X_Total_data = ss.fit_transform(X_Total_data)

X_Total_data

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping
# Build a simple dense model with early stopping and softmax for categorical classification, remember we have 20 classes
model = Sequential()
model.add(Dense(193, input_shape=(193,), activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(20, activation = 'softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
#early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')

history = model.fit(X_Total_data, y_Total_data, batch_size=10, epochs=300) 
                    #validation_data=(X_val, y_val),
                    #callbacks=[early_stop])

#/content/.flac
audio_flac = glob.glob("/content/content/*.flac")
audio_flac

def extract_features(files):
    
    # Sets the name to be the path to where the file is in my computer
    #file_name = os.path.join(os.path.abspath('/content')+'/'+str(files))
    file_name = files
# Loads the audio file as a floating point time series and assigns the default sample rate
    # Sample rate is set to 22050 by default
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
# Generate Mel-frequency cepstral coefficients (MFCCs) from a time series 
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
# Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    stft = np.abs(librosa.stft(X))
# Computes a chromagram from a waveform or power spectrogram.
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
# Computes a mel-scaled spectrogram.
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
# Computes spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
# Computes the tonal centroid features (tonnetz)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    return mfccs, chroma, mel, contrast, tonnetz
count = 0
for d in audio_flac[40:60]:
  #fileName = r"/content/abu-bakr-al-shatri-111-al-masadd-3733-4614.flac"
  testing_features = extract_features(d)
  features_testing = []
  features_testing.append(np.concatenate((
      testing_features[0],
      testing_features[1], 
      testing_features[2], 
      testing_features[3],
      testing_features[4]), axis=0))

  Test_Audio = np.array(features_testing)
  predictions = model.predict_classes(Test_Audio)
  O = d.split("1")[0][17:-1]
  P = str(le.inverse_transform(predictions))[2:-2]
  print("Orignal Speaker is: ",O,"\n")
  print("The pridicted Speaker is :",P)
  if O == P:
    count = count + 1
  #print(predictions)
  print("\n ###########################################################")
print("Total Accuracy on 20 test files: ", count)
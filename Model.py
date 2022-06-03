import os
import pandas as pd
from zipfile import ZipFile
import joblib
import numpy as np
import pydub
import joblib
import librosa
import glob
import warnings
warnings.filterwarnings("ignore")
#file_name2 = "/content/drive/MyDrive/Dataset/Al-Quran App Data/AL Quran App (Voice Recognition)/5 Samples Data/flac_5_Sample.zip"


audioFolder = glob.glob("data_mp3\\*.mp3")

print(audioFolder)
from pydub import AudioSegment
f = "data_mp3//abdallah-kamel-001-al-fatiha-15884-1452.mp3"
# files
#for f in audioFolder:
src = f
dst = str(f[:-3])+"flac"

# convert wav to mp3
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="flac")
print("Done############################################")



import sys
sys.path.append(r"C:\Users\kanoe\anaconda3\Lib\site-packages")

import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os
import datetime as dt
# import o?ceansoundscape  
# from oceansoundscape.spectrogram import conf, denoise, utils
# from oceansoundscape.raven import BLEDParser
from pathlib import Path
import soundfile as sf
import cv2
import json
import numpy as np
# import tensorflow as tf
# import torch
import shutil
from dateutil import parser
import transformers
from transformers import AutoFeatureExtractor
import glob
import scipy.signal as sg
# from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
import tempfile
import pandas as pd
from BledParser import BLEDParser
from datasets import load_dataset
import scipy as sc
from datasets import Dataset,Audio

NUM_EXAMPLES = 30
NUM_LABELS = 2
label2id, id2label = {}, {}
TRAINING_DAYS = ['20150818'] # ,'20160616','20170203']
blue_conf = {'low_freq': 20, 'high_freq': 120, 'duration_secs': 7, 'blur_axis': '', 'num_fft': 1024, 'center': True, 'padding_secs': 2, 'num_mels': 10}
BASE_PATH = r"C:\Users\kanoe\Documents\thesis\ML project"

def download_wav(utcstring,bled_path:Path, num_rows):
    """
    Grab a wav file for a given utc string, e.g. 20200711T000000Z
    """
    date = parser.parse(utcstring)
    wav_filename = f'MARS-{date:%Y%m%dT%H%M%SZ}-16kHz.wav'
    bucket = 'pacific-sound-16khz'
    key = f'{date.year:04d}/{date.month:02d}/{wav_filename}'
    
    s3client = boto3.client(
        's3',
        region_name='us-west-1')

    s3 = boto3.resource('s3',
        aws_access_key_id='',
        aws_secret_access_key='',
        config=Config(signature_version=UNSIGNED))
    
    if not Path(wav_filename).exists():

      print(f'Downloading {key} from s3://{bucket}') 
      s3.Bucket(bucket).download_file(key, wav_filename)
      print('Done')

    num = cache_file(Path(wav_filename),bled_path,utcstring,num_rows)
    # temp = tempfile.NamedTemporaryFile(mode='w+b')
    # with temp as f:
    #     s3.Bucket(bucket).download_file(key, wav_filename,f)
    #     num = cache_file(temp,bled_path,dataset,start)
    # temp.close()

    return num

def cache_file(wav_path, bled_path:Path, date, num_rows):
    # read the wav file
    xc, sample_rate = sf.read(wav_path.name)

    # parse the detections
    parser = BLEDParser(bled_path, blue_conf, len(xc), sample_rate)
    call_width = int(blue_conf['duration_secs'] * sample_rate)
    max_width = len(xc) - 1
    detections_df = parser.data
    path_to_data = os.path.join(BASE_PATH, r"dataset\data\train")

    for row, item in sorted(detections_df.iterrows()):
        start = int(item.call_start - call_width)
        end = int(item.call_end + call_width)
        if start > 0 and end < max_width and (num_rows+row) < NUM_EXAMPLES and item.has_label:
            data = xc[start:end]
            b, a = sg.butter(4, 200/(fr / 2.), 'low')
            x_fil = sg.filtfilt(b, a, data)
            number_of_samples = round(len(x_fil) / 4)
            data = sg.resample(x_fil, number_of_samples)

            if item.classification=='bdt':
                sc.io.wavfile.write(os.path.join(path_to_data,r'\true',str(date[:8]) + '-' + str(item.Selection),'.wav'), sample_rate, data)

            elif item.classification=='bdf':
                sc.io.wavfile.write(os.path.join(path_to_data,r'\true',str(date[:8]) + '-' + str(item.Selection),'.wav'), sample_rate, data)

    return np.shape(detections_df)[0]

# needs a folder called dataset with test/eval/train splits then true/false splits
def loadVariables():
    bled_files = []
    num_rows = 0

    for i in range(len(TRAINING_DAYS)):
        date=TRAINING_DAYS[i]
        bled_file = os.path.join(BASE_PATH,r'\training\MARS-' + date + 'T000000Z-2kHz.wav.selections.txt')
        
        bled_files.append(bled_file)
        num_rows += download_wav(date + "T000000Z",Path(bled_file),num_rows)


def preprocess(audio):
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    audio_arrays = [x["array"] for x in audio["audio"]]

    inputs = feature_extractor(
        audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
    )
    return inputs


def main(): 
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    loadVariables()

    audio_dataset = load_dataset("audiofolder", data_dir=BASE_PATH + "dataset")
    encoded_data = audio_dataset.map(preprocess, remove_columns="audio", batched=True)

    # extract labels from the loaded dataset
    labels = audio_dataset['train'].features['label'].names
    # make the label dict based on labels in dataset
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    
    model = AutoModelForAudioClassification.from_pretrained(
        "facebook/wav2vec2-base", num_labels=NUM_LABELS, label2id=label2id, id2label=id2label
    )

    training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    num_train_epochs=5,
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_data['train'],
    # need validation set build
    eval_dataset=encoded_data['eval'],
    tokenizer=feature_extractor,
    )

    trainer.train()


if __name__ == "__main__":
    main()
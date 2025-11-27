import os
import glob
import librosa.feature as lf
import librosa
import numpy as np
import soundfile as sf
from moviepy import VideoFileClip


def convert_video_to_audio_moviepy(video_file, output_ext="mp3"):
    """Converts video to audio using MoviePy library
    that uses `ffmpeg` under the hood"""
    if not os.path.exists('audio_files'):
        os.makedirs('audio_files')
    filename, ext = os.path.splitext(video_file)
    filename = filename.split("/")[-1]
    clip = VideoFileClip(video_file)
    print("clip",clip)
    # clip.audio.write_audiofile(f"audio_files/{filename}.{output_ext}")
    data = clip.audio.to_soundarray()
    print("data shape",data.shape)
    return

def extract_audio_features():
    for audio_file in glob.glob("audio_files/*.mp3"):
        print(f'Extracting audio features for {audio_file}')
        y, fs = sf.read(audio_file, dtype='float32')
        print("len(y):", len(y))
        print("duration (s):", len(y) / fs)
        print('fs', fs)
        if y.ndim == 2:  # (n_samples, n_channels)
            y = y.T  # -> (n_channels, n_samples)
            y = librosa.to_mono(y)
        print('mono shape:', y.shape)
        print('duration (s):', len(y) / fs)

        mfcc = lf.mfcc(y=y, n_mfcc=20)
        # mfcc = mfcc.squeeze()
        # mfcc = mfcc.T
        print('mfcc', mfcc.shape, mfcc.ndim, type(mfcc))
        break


for file in sorted(glob.glob('audio_video_files/*.MOV')):
    print(f'file: {file}')
    convert_video_to_audio_moviepy(file)
    extract_audio_features()
    break

# print('hello')

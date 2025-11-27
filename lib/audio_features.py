import os
import glob
import librosa.feature as lf
import numpy as np
import soundfile as sf
from moviepy import VideoFileClip


class AudioFeatures:
    def __init__(self):
        pass

    def convert_video_to_audio_moviepy(self, video_file, name, output_ext="mp3"):
        """Converts video to audio using MoviePy library
        that uses `ffmpeg` under the hood"""
        clip = VideoFileClip(video_file)
        data = clip.audio.to_soundarray(fps=44100)
        return data

    def extract_audio_features(self, audio_bytes, fps=44100):
        mfcc = lf.mfcc(y=audio_bytes, n_mfcc=20)
        print('mfcc', mfcc.shape)
        return mfcc

    def get_features(self, path, name):
        # for file in sorted(glob.glob('audio_video_files/*.MOV')):
        #     print(f'file: {file}')
        audio_bytes = self.convert_video_to_audio_moviepy(path, name)
        return self.extract_audio_features(audio_bytes)

    # print('hello')

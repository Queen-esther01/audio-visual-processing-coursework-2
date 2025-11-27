import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import cv2

# use cascasge to crop the lips
# Use ffmeg to recompose the video and work on it.
# then apply dct on a 20ms frames
# compare the result with the audio and visual sector

from lip_extraction import LipExtraction
from utils import Utils



def main():
    print("Hello from audio-visual-processing-coursework-2!")
    # extract_frames()
    lE = LipExtraction()
    data = []
    limit = 0
    for video_file in sorted(glob.glob('audio_video_files/*.MOV')):
        filename = os.path.basename(video_file)
        name, ext = os.path.splitext(filename)
        print("Processing:", name)

        if limit > 0:
            break
        limit += 1

        frames = lE.extract_frames(video_file)
        frame_index = 0
        for frame in frames:
            if frame_index >0:
                break
            frame_index += 1

            print("Test frame", frame)

            s_roi = lE.get_roi(frame, name, draw_plot=False)

            # features = apply_dct(s_roi)
            # data.append(features)


# def main():
#     print("Hello from audio-visual-processing-coursework-2!")
#     # extract_frames()
#     lE = LipExtraction()
#     data = []
#     limit = 0
#     for video_file in sorted(glob.glob('audio_video_files/*.MOV')):
#         if limit > 0:
#             break
#         limit += 1
#         filename = os.path.basename(video_file)
#         name, ext = os.path.splitext(filename)
#         frames = lE.extract_frames(video_file)
#         data.append((frames, name))
#     print("Data length", len(data))
#     for i in range(len(data)):
#
#         test_frame, name = data[i]
#         print("Test frame", test_frame)
#         test_frame = test_frame[0]
#         print("Processing:", name)
#         s_roi = lE.get_roi(test_frame, name, draw_plot=False)
#
#         apply_dct(s_roi)
#
#     # data = extract_frames()
#     # # test_frame, name = data[0]
#     # # test_frame = test_frame[0]
#     # # plt.imshow(cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB))
#     # # plt.show()
#     #
#     # for i in range(len(data)):
#     #     test_frame, name = data[i]
#     #     test_frame = test_frame[0]
#     #     print("Processing:", name)
#     #     get_roi(test_frame, name)
#     #     # apply_dct(test_frame)

if __name__ == "__main__":
    main()

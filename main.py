import os
import glob
from lib.linear_model import LinearModel
from lib.visual_feature_interp import *
from concurrent.futures import ProcessPoolExecutor

# use cascasge to crop the lips
# Use ffmeg to recompose the video and work on it.
# then apply dct on a 20ms frames
# compare the results with the audio and visual sector


base_path = "results"
normalizer_name = f"{base_path}/norm_stats.npz"
output_name = f"{base_path}/correct_predictions.png"


def get_name(file):
    filename = os.path.basename(file)

    name, ext = os.path.splitext(filename)


    name_split = name.split("_")
    label = name_split[0].title()
    label_number = int(name_split[1])
    return name, label, label_number, filename


# def get_features():
#     data = []
#     labels = []
#     limit = 0
#     last_name = ''
#     for file in sorted(glob.glob('audio_video_files/*.MOV')):
#         name, label, label_number, filename = get_name(file)
#
#         if last_name == label and label_number > 6:
#             continue
#         print("Processing:", name)
#
#         last_name = label
#         video_features = vF.get_features(file, filename)
#         # audio_features = aF.get_features(file, name)
#         if video_features is None:
#             # if video_features is None or audio_features is None:
#             continue
#         data.append(video_features)
#
#         labels.append(label)
#     return data, labels


def process_single_file(file):
    from lib.video_features import VideoFeatures
    from lib.audio_features import AudioFeatures
    name, label, label_number, filename = get_name(file)
    print("processing file:", file)

    vf = VideoFeatures()
    af = AudioFeatures()
    video_features = vf.get_features(file, filename)
    audio_features = af.get_features(file, filename)
    combined_features = visual_feature_interp(video_features, audio_features)

    if combined_features is None:
        return None

    return combined_features, label

def get_features():
    all_files = sorted(glob.glob('src/audio_video_files/*.MOV'))

    selected_files = []
    seen_per_label = {}

    for file in all_files:
        name, label, label_number, filename = get_name(file)

        count = seen_per_label.get(label, 0)
        if count >= 6:
            continue

        seen_per_label[label] = count + 1
        selected_files.append(file)

    print("Selected files:", len(selected_files))

    data = []
    labels = []

    with ProcessPoolExecutor(max_workers=4) as executor:
        for result in executor.map(process_single_file, selected_files):
            if result is None:
                continue
            video_features, label = result
            data.append(video_features)
            labels.append(label)

    return data, labels

def main():
    print("Hello from audio-visual-processing-coursework-2!")
    data, labels = get_features()
    from collections import Counter

    print("Label counts:", Counter(labels))
    data = np.array(data)
    linear_model = LinearModel()
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) + 1e-8
    np.savez(normalizer_name, mean=mean, std=std)
    data = (data - mean) / std
    labels = linear_model.labelEncoder(labels)
    linear_model.train_model(data, labels)


if __name__ == "__main__":
    main()

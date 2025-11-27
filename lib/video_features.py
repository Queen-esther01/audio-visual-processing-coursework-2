import numpy as np
import cv2

from lib.lip_extraction import LipExtraction

class VideoFeatures:
    def __init__(self):
        self.lE = LipExtraction()

    def apply_dct(self, s_image):
        img = cv2.cvtColor(s_image, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, (64, 64))

        img_float = np.float32(img)
        img_dct = cv2.dct(img_float)
        img_dct_log = np.log(np.abs(img_dct))

        N = 32
        img_dct_trunc = np.zeros_like(img_dct)
        img_dct_trunc[:N, :N] = img_dct[:N, :N]
        img_new = cv2.idct(img_dct_trunc)
        features = img_new.flatten()
        # Utils.draw_tmp(img, img_new)
        return features


    def features_by_video(self, frames, name):
        frame_feats = []
        frame_index = 0
        for frame in frames:
            # if frame_index >0:
            #     break
            # frame_index += 1

            s_roi = self.lE.get_roi(frame, name, draw_plot=False)
            if s_roi is None:
                continue

            features = self.apply_dct(s_roi)
            # print("features", len(features))
            # data.append(features)

            frame_feats.append(features)
        if not frame_feats:
            return None

        frame_feats = np.array(frame_feats)
        video_features = np.mean(frame_feats, axis=0)
        return video_features

    def get_features(self, path, name):

        frames = self.lE.extract_frames(path)
        video_features = self.features_by_video(frames, name)
        return video_features

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 16:10:05 2025

@author: becca
"""

import numpy as np
import cv2
from utils import Utils


class LipExtraction:

    def extract_frames(self, video_file):
        cap = cv2.VideoCapture(video_file)
        frame_index = 0
        frames = []

        while True:
            success, frame = cap.read()
            if not success:
                break
            # if len(frames) == 1:
            #     break

            # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frames.append(frame)
            frame_index += 1

        cap.release()
        frames_np = np.array(frames)
        return frames_np

    def extract_lip_cords(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Load the Haar cascade for mouth detection
        mouth_cascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")

        if mouth_cascade.empty():
            print("Cascade failed to load. Check path.")
            return None
        else:
            mouths = mouth_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            return self.find_best_cord(mouths)

    def find_best_cord(self, mouths):
        filtered_range = [(x, y, w, h) for (x, y, w, h) in mouths if 300 <= x <= 499 and 1180 <= y]
        if filtered_range:
            focused_filtered_range = [(x, y, w, h) for (x, y, w, h) in mouths if 400 <= x <= 499]
            if focused_filtered_range:
                if len(focused_filtered_range) >= 2:
                    cords = self.optimized_cords(focused_filtered_range)
                else:
                    cords = focused_filtered_range[0]
            else:
                cords = self.optimized_cords(filtered_range)
        else:
            cords = self.optimized_cords(mouths)

        return cords

    def optimized_cords(self, mouth_range):
        print("Finding best cord", len(mouth_range))
        xs = [x for (x, y, w, h) in mouth_range]
        ys = [y for (x, y, w, h) in mouth_range]
        xws = [x + w for (x, y, w, h) in mouth_range]
        yhs = [y + h for (x, y, w, h) in mouth_range]
        x_min = min(xs) - 10
        y_min = min(ys) - 25
        x_max = max(xws) + 50
        y_max = max(yhs) - 80

        cords = (x_min, y_min, x_max - x_min, y_max - y_min)

        return cords

    def get_roi(self, test_frame, filename, draw_plot=False):
        cords = self.extract_lip_cords(test_frame)
        if not cords:
            print("no arg")
            return

        x, y, w, h = cords
        print(x, y, w, h)

        x, y, w, h = cords
        y1, y2 = y, y + h
        x1, x2 = x, x + w
        roi = test_frame[y1:y2, x1:x2]

        segmented_roi, result, segmentation_roi = self.thresholding(roi)
        if draw_plot:
            Utils.draw_plot(roi, segmented_roi, result, segmentation_roi, test_frame, filename + f"{cords}")

        return segmented_roi

    def thresholding(self, roi):

        # Compute mean color of the ROI
        roi_mean_rgb = (
            np.mean(roi[:, :, 0]),
            np.mean(roi[:, :, 1]),
            np.mean(roi[:, :, 2]))

        # Compute distance from mean only inside ROI

        d_squares = (roi - roi_mean_rgb) ** 2
        d_sum = d_squares[:, :, 0] + d_squares[:, :, 1] + d_squares[:, :, 2]
        d = np.sqrt(d_sum)

        # Threshold based on std within ROI
        thresh = np.std(d)
        segmentation_roi = d < thresh

        # Applying segmentation mask to the ROI itself
        segmented_roi = cv2.bitwise_and(roi, roi, mask=segmentation_roi.astype(np.uint8) * 255)

        # hsv segmentation
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_pink = np.array([140, 40, 70])  # H, S, V
        upper_pink = np.array([179, 255, 255])
        mask = cv2.inRange(hsv, lower_pink, upper_pink)
        white_bg = np.full_like(roi, 255)
        black_mouth = np.zeros_like(roi)
        result = np.where(mask[..., None] == 255, black_mouth, white_bg)

        return segmented_roi, result, segmentation_roi


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 16:10:05 2025

@author: becca
"""

import matplotlib.pyplot as plt
import cv2
import matplotlib.gridspec as gridspec


class Utils:
    def __init__(self):
        pass

    @staticmethod
    def draw_plot(roi, segmented_roi, mouth_region, segmentation_gray, test_frame, filename):
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(3, 2, width_ratios=[2, 1], height_ratios=[1, 1, 1])  # 2 rows, 2 cols; left wider

        # Full frame (spans all 3 rows in left column)
        ax0 = fig.add_subplot(gs[0:2, 0])
        ax0.imshow(cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB))
        ax0.set_title(f"Full Frame- {filename}")
        ax0.axis("off")

        # ROI
        ax0 = fig.add_subplot(gs[2, 0])

        ax0.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        ax0.set_title("ROI")
        ax0.axis("off")

        # Segmented ROI
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(segmented_roi, cmap='gray')
        ax2.set_title("Segmented ROI")
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[1, 1])
        ax3.imshow(segmentation_gray, cmap='gray')
        ax3.set_title("Segmented ROI (Gray)")
        ax3.axis("off")

        ax1 = fig.add_subplot(gs[2, 1])
        ax1.imshow(cv2.cvtColor(mouth_region, cv2.COLOR_BGR2RGB))
        ax1.set_title(f"Mouth Frame")
        ax1.axis("off")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def draw_tmp(before, after):
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], height_ratios=[1])  # 2 rows, 2 cols; left wider

        ax0 = fig.add_subplot(gs[:, 0])
        ax0.imshow(cv2.cvtColor(before, cv2.COLOR_BGR2RGB))
        ax0.set_title(f"Before Frame")
        ax0.axis("off")

        # Segmented ROI
        ax2 = fig.add_subplot(gs[:, 1])
        ax2.imshow(after, cmap='gray')
        ax2.set_title("After Frame")
        ax2.axis("off")

        plt.tight_layout()
        plt.show()

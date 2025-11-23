import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import cv2


def extract_frames():
    os.makedirs("result", exist_ok=True)

    data = []
    limit = 0

    for video_file in sorted(glob.glob('audio_video_files/*.MOV')):
        filename = os.path.basename(video_file)
        name, ext = os.path.splitext(filename)
        if limit > 3:
            break
        limit += 1
        cap = cv2.VideoCapture(video_file)
        frame_index = 0

        frames = []

        while True:
            success, frame = cap.read()
            if not success:
                break
            if len(frames) == 1:
                break

            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frames.append(frame)

            frame_index += 1

        cap.release()
        frames_np = np.array(frames)
        data.append((frames_np, name))
    print("Done")
    return data


def extract_lip_cords(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the Haar cascade for mouth detection
    mouth_cascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")

    if mouth_cascade.empty():
        print("Cascade failed to load. Check path.")
    else:
        mouths = mouth_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        filtered_range = [(x, y, w, h) for (x, y, w, h) in mouths if 300 <= x <= 499]
        cords = ()
        if filtered_range:
            fou_filtered_range = [(x, y, w, h) for (x, y, w, h) in mouths if 400 <= x <= 499]

            if len(fou_filtered_range) >= 2:
                # --- NEW: build one bounding box that covers all candidates ---
                xs = [x for (x, y, w, h) in fou_filtered_range]
                ys = [y for (x, y, w, h) in fou_filtered_range]
                xws = [x + w for (x, y, w, h) in fou_filtered_range]
                yhs = [y + h for (x, y, w, h) in fou_filtered_range]

                x_min = min(xs) -10
                y_min = min(ys)-25 
                x_max = max(xws) +50
                y_max = max(yhs)
                cords = (x_min, y_min, x_max - x_min, y_max - y_min)
            else:
                cords = max(filtered_range, key=lambda m: m[0])
        return cords



def get_roi(test_frame, filename):

    cords = extract_lip_cords(test_frame)
    if not cords:
        print("no arg")
        return
    x, y, w, h = cords
    y1, y2 = y, y + h
    x1, x2 = x, x + w
    roi = test_frame[y1:y2, x1:x2]
    roi_mean_rgb = (
        np.mean(roi[:, :, 0]),
        np.mean(roi[:, :, 1]),
        np.mean(roi[:, :, 2]))
    # print("Mean RGB:", roi_mean_rgb)
    d_squares = (test_frame - roi_mean_rgb) ** 2
    d_sum = d_squares[:, :, 0] + d_squares[:, :, 1] + d_squares[:, :, 2]
    d = np.sqrt(d_sum)

    roi = test_frame[y1:y2, x1:x2]

    # Compute mean color of the ROI
    roi_mean_rgb = np.mean(roi.reshape(-1, 3), axis=0)

    # Compute distance from mean only inside ROI
    d = np.sqrt(np.sum((roi - roi_mean_rgb) ** 2, axis=2))

    # Threshold based on std within ROI
    thresh = np.std(d)
    segmentation_roi = d < thresh

    # Applying segmentation mask to the ROI itself
    segmented_roi = cv2.bitwise_and(roi, roi, mask=segmentation_roi.astype(np.uint8) * 255)
    # plt.imshow(cv2.cvtColor(segmented_roi, cv2.COLOR_BGR2RGB))
    # plt.show()
    segmentation_gray = (segmentation_roi * 255).astype(np.uint8)
    draw_plot(roi, segmented_roi, segmentation_gray, test_frame, filename)


def draw_plot(roi, segmented_roi, segmentation_gray, test_frame, filename):
    import cv2
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(3, 2, width_ratios=[2, 1], height_ratios=[1, 1, 1])  # 2 rows, 2 cols; left wider

    # Full frame (spans all 3 rows in left column)
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB))
    ax0.set_title(f"Full Frame- {filename}")
    ax0.axis("off")

    # ROI
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    ax1.set_title("ROI")
    ax1.axis("off")

    # Segmented ROI
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.imshow(segmented_roi, cmap='gray')
    ax2.set_title("Segmented ROI")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[2, 1])
    ax3.imshow(segmentation_gray, cmap='gray')
    ax3.set_title("Segmented ROI (Gray)")
    ax3.axis("off")

    plt.tight_layout()
    plt.show()






def main():
    print("Hello from audio-visual-processing-coursework-2!")
    # extract_frames()

    data = extract_frames()
    # test_frame, name = data[0]
    # test_frame = test_frame[0]
    # plt.imshow(cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB))
    # plt.show()

    for i in range(len(data)):
        test_frame, name = data[i]
        test_frame = test_frame[0]
        print("Processing:", name)
        get_roi(test_frame, name)

if __name__ == "__main__":
    main()

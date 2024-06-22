import cv2
from skimage.transform import resize
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")


EMPTY = True
NOT_EMPTY = False

MODEL = pickle.load(open("xgb_model_new.pkl", "rb"))


def empty_or_not(spot_bgr):

    flat_data = []
    img_resized = resize(spot_bgr, (15, 15))
    # img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    y_output = MODEL.predict(flat_data)

    if y_output == 0:
        return EMPTY
    else:
        return NOT_EMPTY


def get_parking_spots_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1
    for i in range(1, totalLabels):

        # Now extract the coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])

    return slots


def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))


def get_video_and_mask():
    video_choice = input("Enter the video number (1, 2, or 3): ")
    if video_choice not in ['1', '2', '3']:
        print("Invalid choice. Please enter 1, 2, or 3.")
        return get_video_and_mask()

    mask_path = f'data/parking_spot_box{video_choice}.png'
    video_path = f'data/sourcevid{video_choice}.mp4'

    return mask_path, video_path

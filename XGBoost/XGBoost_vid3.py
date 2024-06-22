import random
import cv2
import numpy as np
from util import get_parking_spots_bboxes, empty_or_not, calc_diff

mask_path = './Mask/parking_spot_box3.png'
video_path = './TestVid/sourcevid3.mp4'

mask = cv2.imread(mask_path, 0)
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
if not ret:
    print("Failed to read the video")
    exit(1)

mask_resized = cv2.resize(
    mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

connected_components = cv2.connectedComponentsWithStats(
    mask_resized, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)

spots_status = [None for _ in spots]
diffs = [None for _ in spots]
scores = [None for _ in spots]  # To store the static scores for each spot

previous_frame = None
frame_nmr = 0
step = 30

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            diffs[spot_indx] = calc_diff(
                spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

        print([diffs[j] for j in np.argsort(diffs)][::-1])

    if frame_nmr % step == 0:
        if previous_frame is None:
            arr_ = range(len(spots))
        else:
            arr_ = [j for j in np.argsort(
                diffs) if diffs[j] / np.amax(diffs) > 0.4]
        for spot_indx in arr_:
            spot = spots[spot_indx]
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            spot_status = empty_or_not(spot_crop)

            # Update the score only if the spot status changes
            if spots_status[spot_indx] != spot_status:
                spots_status[spot_indx] = spot_status
                if spot_status:
                    scores[spot_indx] = - \
                        random.uniform(0, 1)  # Random score < 0
                else:
                    scores[spot_indx] = random.uniform(
                        0, 1)  # Random score > 0

    if frame_nmr % step == 0:
        previous_frame = frame.copy()

    for spot_indx, spot in enumerate(spots):
        spot_status = spots_status[spot_indx]
        score = scores[spot_indx]
        x1, y1, w, h = spots[spot_indx]

        if spot_status:
            color = (0, 255, 0)  # Green
        else:
            color = (0, 0, 255)  # Red

        frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)
        cv2.putText(frame, f'{score:.2f}', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, f'Available spots: {sum(spots_status)} / {len(spots_status)}', (100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()

#Manually Count
true_positives = 16
true_negatives = 151
false_positives = 70
false_negatives = 9

accuracy = (true_positives + true_negatives) /  (true_positives + true_negatives + false_positives + false_negatives)
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)
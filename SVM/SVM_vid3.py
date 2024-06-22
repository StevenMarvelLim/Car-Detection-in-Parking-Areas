import cv2
import numpy as np
import pickle
from skimage.transform import resize

IS_EMPTY = True
IS_NOT_EMPTY = False

model_path = "./Project Car Detection SVM/model.p"
with open(model_path, 'rb') as file:
    PARKING_MODEL = pickle.load(file)

def calculate_difference(image1, image2):
    return np.abs(np.mean(image1) - np.mean(image2))

def is_empty_spot(spot_image):
    flattened_data = []
    resized_image = resize(spot_image, (15, 15, 3))
    flattened_data.append(resized_image.flatten())
    flattened_data = np.array(flattened_data)
    decision_scores = PARKING_MODEL.decision_function(flattened_data)
    prediction_output = PARKING_MODEL.predict(flattened_data)
    confidence = decision_scores.max()
    decision_scores *= -1
    if prediction_output == 0:
        return IS_EMPTY, confidence
    else:
        return IS_NOT_EMPTY, confidence

def extract_parking_spots(connected_components):
    (total_labels, label_ids, stats, centroids) = connected_components
    parking_slots = []
    for i in range(1, total_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        width = int(stats[i, cv2.CC_STAT_WIDTH])
        height = int(stats[i, cv2.CC_STAT_HEIGHT])
        parking_slots.append([x, y, width, height])
    return parking_slots

mask_filepath = './Mask/parking_spot_box3.png'
video_filepath = './TestVid/sourcevid3.mp4'

mask_image = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
video_capture = cv2.VideoCapture(video_filepath)

ret, initial_frame = video_capture.read()
if not ret:
    print("Error reading video file")
    video_capture.release()
    cv2.destroyAllWindows()
    exit()

frame_height, frame_width = initial_frame.shape[:2]

resized_mask = cv2.resize(mask_image, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

connected_components = cv2.connectedComponentsWithStats(resized_mask, 4, cv2.CV_32S)
parking_spots = extract_parking_spots(connected_components)

spot_status_list = [None for _ in parking_spots]
spot_confidence_list = [None for _ in parking_spots]
difference_list = [None for _ in parking_spots]

previous_frame = None
frame_number = 0
ret = True
frame_step = 30

while ret:
    ret, current_frame = video_capture.read()
    if not ret:
        break

    if frame_number % frame_step == 0 and previous_frame is not None:
        for spot_index, spot in enumerate(parking_spots):
            x, y, width, height = spot
            spot_crop = current_frame[y:y + height, x:x + width, :]
            difference_list[spot_index] = calculate_difference(spot_crop, previous_frame[y:y + height, x:x + width, :])
        
    if frame_number % frame_step == 0:
        if previous_frame is None:
            spot_indices_to_check = range(len(parking_spots))
        else:
            spot_indices_to_check = [j for j in np.argsort(difference_list) if difference_list[j] / np.amax(difference_list) > 0.4]
        for spot_index in spot_indices_to_check:
            spot = parking_spots[spot_index]
            x, y, width, height = spot
            spot_crop = current_frame[y:y + height, x:x + width, :]
            spot_status, confidence = is_empty_spot(spot_crop)
            spot_status_list[spot_index] = spot_status
            spot_confidence_list[spot_index] = confidence
            
    if frame_number % frame_step == 0:
        previous_frame = current_frame.copy()
        
    for spot_index, spot in enumerate(parking_spots):
        spot_status = spot_status_list[spot_index]
        confidence = spot_confidence_list[spot_index]
        x, y, width, height = parking_spots[spot_index]
        color = (0, 255, 0) if spot_status else (0, 0, 255)
        current_frame = cv2.rectangle(current_frame, (x, y), (x + width, y + height), color, 2)
        font_scale = min(width, height) / 60
        cv2.putText(current_frame, f'{confidence:.2f}', (x, y - int(10 * font_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

    cv2.rectangle(current_frame, (80, 20), (550, 80), (0, 0, 0), -1)
    
    cv2.putText(current_frame, 'Available spots: {} / {}'.format(str(sum(spot_status_list)), str(len(spot_status_list))), (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.namedWindow('Parking Spots', cv2.WINDOW_NORMAL)
    cv2.imshow('Parking Spots', current_frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    frame_number += 1

video_capture.release()
cv2.destroyAllWindows()

#Manually Count
true_positives = 17
true_negatives = 127
false_positives = 69
false_negatives = 10

accuracy = (true_positives + true_negatives) / len(parking_spots)
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)
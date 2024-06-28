import cv2
import tensorflow as tf
from skimage.transform import resize
import numpy as np


IS_EMPTY = True
IS_NOT_EMPTY = False

# Define MobileNetV2 base model
MbNetV2_base = tf.keras.applications.MobileNetV2(input_shape=(32, 32, 3),
                                                 include_top=False,
                                                 weights=None)

# Add custom classification layers on top of MobileNetV2
x = MbNetV2_base.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
predictions = tf.keras.layers.Dense(2, activation='softmax')(x)
MbNetV2 = tf.keras.Model(inputs=MbNetV2_base.input, outputs=predictions)

# Load weights (entire model state)
MbNetV2.load_weights('./Project Car Detection MobileNetV2/MbNetV2.weights.h5')

# Example function to use the loaded model
def is_empty_spot(spot_image):
    label_names = ['empty', 'not_empty']
    
    # Resize image to match model input shape
    resized_image = resize(spot_image, (32, 32))
    
    # Convert to TensorFlow tensor and expand dimensions
    resized_image = tf.expand_dims(tf.convert_to_tensor(np.array(resized_image), dtype=tf.float32), 0)
    
    # Make prediction
    decision_scores = MbNetV2(resized_image)
    prediction_output = np.argmax(decision_scores.numpy())
    confidence = label_names[prediction_output]
    
    if prediction_output == 0:
        return IS_EMPTY, confidence
    else:
        return IS_NOT_EMPTY, confidence
    
def calculate_difference(image1, image2):
    return np.abs(np.mean(image1) - np.mean(image2))

# Function to extract parking spots from connected components
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

# Initialize paths for mask and video files
mask_filepath = './Mask/parking_spot_box1.png'
video_filepath = './TestVid/sourcevid1.mp4'

# Read and process mask image and video capture
mask_image = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
video_capture = cv2.VideoCapture(video_filepath)

# Initialize variables for frame dimensions and resizing mask
ret, initial_frame = video_capture.read()
if not ret:
    print("Error reading video file")
    video_capture.release()
    cv2.destroyAllWindows()
    exit()

frame_height, frame_width = initial_frame.shape[:2]
resized_mask = cv2.resize(mask_image, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

# Obtain connected components from resized mask image
connected_components = cv2.connectedComponentsWithStats(resized_mask, 4, cv2.CV_32S)
parking_spots = extract_parking_spots(connected_components)

# Initialize lists for spot status, confidence, and difference
spot_status_list = [None for _ in parking_spots]
spot_confidence_list = [None for _ in parking_spots]
difference_list = [None for _ in parking_spots]

# Initialize variables for previous frame and frame number
previous_frame = None
frame_number = 0
ret = True
frame_step = 120

# Loop through video frames for processing
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
        cv2.putText(current_frame, f'{confidence}', (x, y - int(10 * font_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

    # Display frame with parking spots information
    cv2.rectangle(current_frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(current_frame, 'Available spots: {} / {}'.format(str(sum(spot_status_list)), str(len(spot_status_list))), (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.namedWindow('Parking Spots', cv2.WINDOW_NORMAL)
    cv2.imshow('Parking Spots', current_frame)
    
    # Wait for 'q' key press to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    frame_number += 1

# Release video capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
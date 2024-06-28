import cv2
import numpy as np
import PIL.Image as img
import tensorflow as tf
from tqdm import tqdm
from keras import layers
from skimage.transform import resize

IS_EMPTY = True
IS_NOT_EMPTY = False

# Models 
class VGG3(tf.keras.Model): 
    def __init__(self,n_class , *args, **kwargs):
        super(VGG3 , self).__init__(*args, **kwargs)
        self.La1 = tf.keras.Sequential(layers=[
            layers.Conv2D(32 , (3,3) , padding='same'), 
            layers.ReLU(), 
            layers.BatchNormalization(axis = -1),
            layers.MaxPooling2D((2,2)), 
            layers.Dropout(0.25)
        ])
        self.flat = layers.Flatten()
        self.linear1 = layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.0005))
        self.linear2 = layers.Dropout(0.25)
        self.Rel = layers.ReLU()
        self.soft = layers.Softmax()
        self.out = layers.Dense(n_class)
        self.bcn = layers.LayerNormalization()
    def call(self, inputs):
        x = self.La1(inputs)
        x = self.flat(x)
        x = self.linear1(x)
        x = self.Rel(x)
        x = self.bcn(x)
        x = self.linear2(x)
        x = self.out(x)
        x = self.soft(x)
        return x 
        
ModelVGG = VGG3(2)
model_dir = "./Project Car Detection VGG3/VGGw3.weight.h5"
print(ModelVGG(tf.random.normal((1,30,30,3))))
ModelVGG.load_weights(model_dir)

def calculate_difference(image1, image2):
    return np.abs(np.mean(image1) - np.mean(image2))

def is_empty_spot(spot_image):
    l = ['empty' , 'not_empty']
    resized_image = resize(spot_image, (30, 30))
    resized_image2 = tf.expand_dims(
        tf.convert_to_tensor(np.array(resized_image), dtype=tf.float32), 0)
    print(resized_image2)
    decision_scores = ModelVGG(resized_image2)
    prediction_output = np.argmax(decision_scores.numpy())
    confidence = l[prediction_output]
    print(prediction_output , confidence)
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

mask_filepath = './Mask/parking_spot_box2.png'
video_filepath = './TestVid/sourcevid2.mp4'

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
        cv2.putText(current_frame, f'{confidence}', (x, y - int(10 * font_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

    cv2.rectangle(current_frame, (80, 20), (550, 80), (0, 0, 0), -1)
    
    cv2.putText(current_frame, 'Available spots: {} / {}'.format(str(sum(spot_status_list)), str(len(spot_status_list))), (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.namedWindow('Parking Spots', cv2.WINDOW_NORMAL)
    cv2.imshow('Parking Spots', current_frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    frame_number += 1

video_capture.release()
cv2.destroyAllWindows()
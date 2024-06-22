import numpy as np
import cv2
import os
from skimage.transform import resize
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import log_loss
import pickle
import matplotlib.pyplot as plt

# Function to load and preprocess images
def load_and_preprocess_images(image_paths, target_size=(64, 64)):
    images = []
    for path in image_paths:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = resize(image, target_size)  # Resize image
        images.append(image)
    return np.array(images)

# Function to extract features from images
def extract_features(images):
    features = []
    for image in images:
        features.append(image.flatten())
    return np.array(features)

# Load dataset
empty_images_dir = './Dataset/empty/'
not_empty_images_dir = './Dataset/not_empty/'

empty_image_paths = [os.path.join(empty_images_dir, filename) for filename in os.listdir(empty_images_dir)]
not_empty_image_paths = [os.path.join(not_empty_images_dir, filename) for filename in os.listdir(not_empty_images_dir)]

# Load and preprocess images
empty_images = load_and_preprocess_images(empty_image_paths)
not_empty_images = load_and_preprocess_images(not_empty_image_paths)

# Extract features
X_empty = extract_features(empty_images)
X_not_empty = extract_features(not_empty_images)

# Create labels
y_empty = np.zeros(len(X_empty))
y_not_empty = np.ones(len(X_not_empty))

# Concatenate features and labels
X = np.concatenate((X_empty, X_not_empty))
y = np.concatenate((y_empty, y_not_empty))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
svm_model = SVC(kernel='linear', probability=True)

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5)

# Lists to store metrics
train_accuracies = []
test_accuracies = []
val_accuracies = []
train_losses = []
test_losses = []
val_losses = []

for fold_idx, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
    X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
    
    # Further split the training data into training and validation sets
    X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(X_train_fold, y_train_fold, test_size=0.2, random_state=42)
    
    # Train the model
    svm_model.fit(X_train_fold, y_train_fold)
    
    # Predict probabilities for log loss calculation
    y_prob_train = svm_model.predict_proba(X_train_fold)
    y_prob_test = svm_model.predict_proba(X_test_fold)
    y_prob_val = svm_model.predict_proba(X_val_fold)
    
    # Calculate accuracy and log loss
    train_acc = svm_model.score(X_train_fold, y_train_fold)
    test_acc = svm_model.score(X_test_fold, y_test_fold)
    val_acc = svm_model.score(X_val_fold, y_val_fold)
    train_loss = log_loss(y_train_fold, y_prob_train)
    test_loss = log_loss(y_test_fold, y_prob_test)
    val_loss = log_loss(y_val_fold, y_prob_val)
    
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    val_accuracies.append(val_acc)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    val_losses.append(val_loss)

# Evaluate on the test set
test_accuracy = svm_model.score(X_test, y_test)

print("Training Accuracy (mean across folds):", np.mean(train_accuracies))
print("Validation Accuracy (mean across folds):", np.mean(val_accuracies))
print("Testing Accuracy (mean across folds):", np.mean(test_accuracies))
print("Testing Accuracy:", test_accuracy)

# Save model
pickle.dump(svm_model, open("model.p", "wb"))

# Plot metrics
plt.figure(figsize=(18, 6))

# Model Accuracy
plt.subplot(1, 3, 1)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, marker='o', label='Train Accuracy')
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, marker='o', label='Test Accuracy')
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, marker='o', label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()

# Model Loss
plt.subplot(1, 3, 2)
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Train Loss')
plt.plot(range(1, len(test_losses) + 1), test_losses, marker='o', label='Test Loss')
plt.title("Model Loss")
plt.xlabel("Fold")
plt.ylabel("Log Loss")
plt.grid(True)
plt.legend()

# Validation Loss
plt.subplot(1, 3, 3)
plt.plot(range(1, len(val_losses) + 1), val_losses, marker='o', label='Validation Loss')
plt.title("Validation Loss")
plt.xlabel("Fold")
plt.ylabel("Log Loss")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

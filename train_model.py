import os
import csv
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence

# ---------------------- Step 1: Load and Balance Data ----------------------
import os
import csv
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence

# ---------------------- Step 1: Load and Balance Data ----------------------
def load_and_balance_data(csv_file, img_folder, n_bins=21, samples_per_bin=1000):
    print("[INFO] Loading and balancing data...")
    image_paths = []
    steerings = []

    with open(csv_file) as file:
        reader = csv.reader(file)
        for row in reader:
            path = os.path.join(img_folder, os.path.basename(row[0]).strip())
            angle = float(row[3])
            image_paths.append(path)
            steerings.append(angle)

    image_paths = np.array(image_paths)
    steerings = np.array(steerings)

    hist, bins = np.histogram(steerings, n_bins)
    remove_list = []

    for i in range(n_bins):
        bin_indices = [j for j in range(len(steerings)) if bins[i] <= steerings[j] <= bins[i+1]]
        if len(bin_indices) > samples_per_bin:
            bin_indices = random.sample(bin_indices, len(bin_indices) - samples_per_bin)
            remove_list.extend(bin_indices)

    image_paths = np.delete(image_paths, remove_list, axis=0)
    steerings = np.delete(steerings, remove_list, axis=0)

    print(f"[INFO] Balanced dataset: {len(image_paths)} samples remaining")
    return image_paths, steerings

# ---------------------- Step 2: Preprocess Images ----------------------
def preprocess_image(img):
    img = img[60:135, :, :]  # Crop road
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img

# ---------------------- Step 3: Create Generator ----------------------
class DataGenerator(Sequence):
    def __init__(self, image_paths, steerings, batch_size=64, is_training=True):
        self.image_paths = image_paths
        self.steerings = steerings
        self.batch_size = batch_size
        self.is_training = is_training

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):
        batch_paths = self.image_paths[index*self.batch_size:(index+1)*self.batch_size]
        batch_steers = self.steerings[index*self.batch_size:(index+1)*self.batch_size]

        images = []
        angles = []
        for path, angle in zip(batch_paths, batch_steers):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess_image(img)
            images.append(img)
            angles.append(angle)

            # Flip only if the steering is significant to avoid bias
            if self.is_training and abs(angle) > 0.05:
                flipped_img = cv2.flip(img, 1)
                flipped_angle = -angle
                images.append(flipped_img)
                angles.append(flipped_angle)

            # Display original image with angle overlay (only 1st image in batch for visualization)
            if self.is_training and len(images) == 1:
                vis = cv2.resize((img * 255).astype(np.uint8), (400, 132))
                vis = cv2.putText(vis, f"Steering: {angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow('Preview (1st in batch)', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

        return np.array(images), np.array(angles)

# ---------------------- Step 4: Define Model ----------------------
def nvidia_model():
    model = Sequential()
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=(66, 200, 3)))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))  # Output: steering angle
    return model

# ---------------------- Step 5: Train Model ----------------------
def train_model():
    csv_file = 'driving_log.csv'
    img_folder = 'IMG'

    image_paths, steerings = load_and_balance_data(csv_file, img_folder)
    X_train_paths, X_val_paths, y_train, y_val = train_test_split(image_paths, steerings, test_size=0.2, shuffle=True)

    train_gen = DataGenerator(X_train_paths, y_train, batch_size=64, is_training=True)
    val_gen = DataGenerator(X_val_paths, y_val, batch_size=64, is_training=False)

    model = nvidia_model()
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

    print("[INFO] Training model...")
    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=5,
                        callbacks=[checkpoint],
                        verbose=1)

    print("[INFO] Training complete. Best model saved as model.h5")

    # Plot training results
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig("training_plot.png")
    plt.show()

if __name__ == '__main__':
    train_model()


# ---------------------- Step 2: Preprocess Images ----------------------
def preprocess_image(img):
    img = img[60:135, :, :]  # Crop road
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img

# ---------------------- Step 3: Create Generator ----------------------
class DataGenerator(Sequence):
    def __init__(self, image_paths, steerings, batch_size=64, is_training=True):
        self.image_paths = image_paths
        self.steerings = steerings
        self.batch_size = batch_size
        self.is_training = is_training

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):
        batch_paths = self.image_paths[index*self.batch_size:(index+1)*self.batch_size]
        batch_steers = self.steerings[index*self.batch_size:(index+1)*self.batch_size]

        images = []
        angles = []
        for path, angle in zip(batch_paths, batch_steers):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess_image(img)
            images.append(img)
            angles.append(angle)

            # Flip only if the steering is significant to avoid bias
            if self.is_training and abs(angle) > 0.05:
                flipped_img = cv2.flip(img, 1)
                flipped_angle = -angle
                images.append(flipped_img)
                angles.append(flipped_angle)

            # Display original image with angle overlay (only 1st image in batch for visualization)
            if self.is_training and len(images) == 1:
                vis = cv2.resize((img * 255).astype(np.uint8), (400, 132))
                vis = cv2.putText(vis, f"Steering: {angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow('Preview (1st in batch)', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

        return np.array(images), np.array(angles)

# ---------------------- Step 4: Define Model ----------------------
def nvidia_model():
    model = Sequential()
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=(66, 200, 3)))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))  # Output: steering angle
    return model

# ---------------------- Step 5: Train Model ----------------------
def train_model():
    csv_file = 'driving_log.csv'
    img_folder = 'IMG'

    image_paths, steerings = load_and_balance_data(csv_file, img_folder)
    X_train_paths, X_val_paths, y_train, y_val = train_test_split(image_paths, steerings, test_size=0.2, shuffle=True)

    train_gen = DataGenerator(X_train_paths, y_train, batch_size=64, is_training=True)
    val_gen = DataGenerator(X_val_paths, y_val, batch_size=64, is_training=False)

    model = nvidia_model()
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

    print("[INFO] Training model...")
    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=5,
                        callbacks=[checkpoint],
                        verbose=1)

    print("[INFO] Training complete. Best model saved as model.h5")

    # Plot training results
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig("training_plot.png")
    plt.show()

if __name__ == '__main__':
    train_model()

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

# Step 1: Load and Balance Data
def load_and_balance_data(csv_file, img_folder, n_bins=21, samples_per_bin=1000): # 21 bins and 1000 samples per bin
    print("[INFO] Loading and balancing data...") # print info to console
    image_paths = [] # list to store image paths
    steerings = [] # list to store steering angles

    with open(csv_file) as file: # open csv file from the current directory
        reader = csv.reader(file) # read the csv file
        for row in reader: # iterate through the rows of the csv file
            path = os.path.join(img_folder, os.path.basename(row[0]).strip()) # join the image folder from IMG in the current directory with the image path
            angle = float(row[3]) # get the steering angle from the csv file in the 4th column
            image_paths.append(path) # append the image path to the image_paths list
            steerings.append(angle) # append the steering angle to the steerings list

    image_paths = np.array(image_paths)
    steerings = np.array(steerings)

    hist, bins = np.histogram(steerings, n_bins) # histogram of the steering angles
    remove_list = []

    for i in range(n_bins): # iterate through the bins
        bin_indices = [j for j in range(len(steerings)) if bins[i] <= steerings[j] <= bins[i+1]] # return the indices of the steering angles that are in the current bin
        if len(bin_indices) > samples_per_bin: # if the number of steering angles in the current bin is greater than the number of samples per bin 
            bin_indices = random.sample(bin_indices, len(bin_indices) - samples_per_bin) # randomly sample the number of steering angles in the current bin to the number of samples per bin
            remove_list.extend(bin_indices) # extend the remove_list with the indices of the steering angles that are in the current bin

    image_paths = np.delete(image_paths, remove_list, axis=0) 
    steerings = np.delete(steerings, remove_list, axis=0)

    print(f"[INFO] Balanced dataset: {len(image_paths)} samples remaining")
    return image_paths, steerings

# ---------------------- Step 2: Preprocess Images ----------------------
def preprocess_image(img): 
    img = img[60:135, :, :]  # Crop road as required from the PDF 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) # convert the image to YUV color space
    img = cv2.GaussianBlur(img, (3, 3), 0) # apply Gaussian blur to the image
    img = cv2.resize(img, (200, 66)) # resize the image to 200x66 
    img = img / 255.0 # normalize the image
    return img

# ---------------------- Step 3: Create Generator ----------------------
class DataGenerator(Sequence):
    def __init__(self, image_paths, steerings, batch_size=64, is_training=True): 
        self.image_paths = image_paths
        self.steerings = steerings
        self.batch_size = batch_size
        self.is_training = is_training

    def __len__(self):
        return len(self.image_paths) // self.batch_size # return the number of batches in the dataset

    def __getitem__(self, index):
        batch_paths = self.image_paths[index*self.batch_size:(index+1)*self.batch_size] # get the image paths for the current batch 
        batch_steers = self.steerings[index*self.batch_size:(index+1)*self.batch_size] # get the steering angles for the current batch

        images = []
        angles = []
        for path, angle in zip(batch_paths, batch_steers): # iterate through the image paths and steering angles for the current batch
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess_image(img)
            images.append(img)
            angles.append(angle)

            # Flip only if the steering is significant to avoid bias
            if self.is_training and abs(angle) > 0.05: # if the steering is significant to avoid the bias
                flipped_img = cv2.flip(img, 1) # flip the image horizontally
                flipped_angle = -angle # negate the steering angle
                images.append(flipped_img) # append the flipped image to the images list
                angles.append(flipped_angle) # append the flipped steering angle to the angles list

            # Display original image with angle overlay (only 1st image in batch for visualization)
            if self.is_training and len(images) == 1:
                vis = cv2.resize((img * 255).astype(np.uint8), (400, 132)) 
                vis = cv2.putText(vis, f"Steering: {angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow('Preview (1st in batch)', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

        return np.array(images), np.array(angles)

# ---------------------- Step 4: Define Model ----------------------
def nvidia_model(): # define the model as required from the PDF
    model = Sequential()
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=(66, 200, 3))) # add the first convolutional layer
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu')) # add the second convolutional layer
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu')) # add the third convolutional layer
    model.add(Conv2D(64, (3, 3), activation='relu')) # add the fourth convolutional layer
    model.add(Conv2D(64, (3, 3), activation='relu')) # add the fifth convolutional layer
    model.add(Flatten()) # flatten the model
    model.add(Dense(100, activation='relu')) # add the first dense layer
    model.add(Dense(50, activation='relu')) # add the second dense layer
    model.add(Dense(10, activation='relu')) # add the third dense layer
    model.add(Dense(1))  # Output: steering angle
    return model

# ---------------------- Step 5: Train Model ----------------------
def train_model():
    csv_file = 'driving_log.csv'
    img_folder = 'IMG'

    image_paths, steerings = load_and_balance_data(csv_file, img_folder) # load and balance the data
    X_train_paths, X_val_paths, y_train, y_val = train_test_split(image_paths, steerings, test_size=0.2, shuffle=True) # split the data into training and validation sets

    train_gen = DataGenerator(X_train_paths, y_train, batch_size=64, is_training=True) # create the training generator 
    val_gen = DataGenerator(X_val_paths, y_val, batch_size=64, is_training=False) # create the validation generator

    model = nvidia_model()
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse') # compile the model

    checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True) # Save the model name as model.h5

    print("[INFO] Training model...")
    history = model.fit(train_gen, # train the model
                        validation_data=val_gen, # validate the model
                        epochs=5, # number of epochs
                        callbacks=[checkpoint], # save the model
                        verbose=1) # print the progress of the training

    print("[INFO] Training complete. Best model saved as model.h5")

    # Plot training results
    plt.plot(history.history['loss'], label='Train Loss') # plot the training loss
    plt.plot(history.history['val_loss'], label='Validation Loss') # plot the validation loss
    plt.legend() 
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig("training_plot.png")
    plt.show()

if __name__ == '__main__':
    train_model() # train the model


import os
import re
import numpy as np
import cv2
from import_data import create_spectrogram
from slice_spectrogram import slice_spect
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

"""
Converts images and labels into training and testing matrices.
"""
def load_dataset(verbose=0, mode=None, datasetSize=1.0):
    create_spectrogram(verbose, mode)
    slice_spect(verbose, mode)

    # datasetSize is a float value which returns a fraction of the dataset.
    # If set as 1.0 it returns the entire dataset.
    # If set as 0.5 it returns half the dataset.

    if mode=="Train":
        genre = {
        "Hip-Hop": 0,
        "International": 1,
        "Electronic": 2,
        "Folk" : 3,
        "Experimental": 4,
        "Rock": 5,
        "Pop": 6,
        "Instrumental": 7
        }
        if(verbose > 0):
            print("Compiling Training and Testing Sets ...")
        
        # Check if Training_Data already exists, load it if present
        if os.path.exists('Training_Data') and os.path.exists("Training_Data/train_x.npy"):
            print("Loading existing training data...")
            train_x = np.load("Training_Data/train_x.npy")
            train_y = np.load("Training_Data/train_y.npy")
            test_x = np.load("Training_Data/test_x.npy")
            test_y = np.load("Training_Data/test_y.npy")
            genre_new = {value: key for key, value in genre.items()}
            n_classes = len(genre)
            print(f"Loaded data shapes - train_x: {train_x.shape}, train_y: {train_y.shape}")
            return train_x, train_y, test_x, test_y, n_classes, genre_new
        
        # Check if sliced images directory exists
        sliced_dir = "Train_Sliced_Images"
        if not os.path.exists(sliced_dir):
            raise FileNotFoundError(f"Directory {sliced_dir} not found. Make sure the dataset is properly processed.")
            
        # Get file list and check if any files exist
        filenames = [os.path.join(sliced_dir, f) for f in os.listdir(sliced_dir)
                     if f.endswith(".jpg")]
        
        if len(filenames) == 0:
            raise ValueError(f"No image files found in {sliced_dir}. Check if spectrograms were created properly.")
            
        print(f"Found {len(filenames)} training image files")
        
        # Process images and labels
        images_all = []
        labels_all = []
        
        for f in filenames:
            try:
                index = int(re.search('Train_Sliced_Images/(.+?)_.*.jpg', f).group(1))
                genre_variable = re.search('Train_Sliced_Images/.*_(.+?).jpg', f).group(1)
                
                if genre_variable not in genre:
                    print(f"Warning: Unknown genre '{genre_variable}' in file {f}, skipping")
                    continue
                    
                temp = cv2.imread(f, cv2.IMREAD_UNCHANGED)
                if temp is None:
                    print(f"Warning: Could not read image {f}, skipping")
                    continue
                    
                # Dynamically grow arrays instead of pre-allocating
                images_all.append(cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY))
                labels_all.append(genre[genre_variable])
            except Exception as e:
                print(f"Error processing {f}: {e}")
                continue

        if len(images_all) == 0:
            raise ValueError("No valid images could be processed. Check image files and genre labels.")
            
        print(f"Successfully loaded {len(images_all)} images")

        # Convert to numpy arrays
        images = np.array(images_all)
        labels = np.array(labels_all)
        labels = labels.reshape(labels.shape[0], 1)
        
        # Split with minimum size validation
        if len(images) < 20:  # arbitrary small number to ensure splitting works
            test_size = 0.2  # Use 20% for test if dataset is very small
        else:
            test_size = 0.05
            
        train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size=test_size, shuffle=True)
        
        # Convert labels to categorical
        train_y = to_categorical(train_y)
        test_y = to_categorical(test_y, num_classes=8)
        n_classes = len(genre)
        genre_new = {value: key for key, value in genre.items()}

        # Save processed data
        if not os.path.exists('Training_Data'):
            os.makedirs('Training_Data')
            
        np.save("Training_Data/train_x.npy", train_x)
        np.save("Training_Data/train_y.npy", train_y)
        np.save("Training_Data/test_x.npy", test_x)
        np.save("Training_Data/test_y.npy", test_y)
        
        return train_x, train_y, test_x, test_y, n_classes, genre_new

    # Rest of the function remains the same
    if mode=="Test":
  
        if(verbose > 0):
            print("Compiling Training and Testing Sets ...")
        filenames = [os.path.join("Test_Sliced_Images", f) for f in os.listdir("Test_Sliced_Images")
                       if f.endswith(".jpg")]
        images = []
        labels = []
        for f in filenames:
            song_variable = re.search('Test_Sliced_Images/.*_(.+?).jpg', f).group(1)
            tempImg = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            images.append(cv2.cvtColor(tempImg, cv2.COLOR_BGR2GRAY))
            labels.append(song_variable)

        images = np.array(images)

        return images, labels
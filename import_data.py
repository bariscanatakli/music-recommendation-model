import os
import pandas as pd
import re
import math
import numpy as np
from PIL import Image
import librosa
import librosa.display
import matplotlib.pyplot as plt

"""
Convert 30s mp3 files into mel-spectrograms.

A mel-spectrograms is a kind of time-frequency representation.
It is obtained from an audio signal by computing the Fourier transforms of short, overlapping windows.
Each of these Fourier transforms constitutes a frame.
These successive frames are then concatenated into a matrix to form the spectrogram.
"""
def create_spectrogram(verbose=0, mode=None):
    print(f"Starting create_spectrogram in {mode} mode")
    
    if mode == "Train":
        # Delete empty directories to force regeneration
        if os.path.exists('Train_Spectogram_Images') and len(os.listdir('Train_Spectogram_Images')) == 0:
            print("Removing empty Train_Spectogram_Images directory")
            os.rmdir('Train_Spectogram_Images')
        
        if os.path.exists('Train_Sliced_Images') and len(os.listdir('Train_Sliced_Images')) == 0:
            print("Removing empty Train_Sliced_Images directory")
            os.rmdir('Train_Sliced_Images')
            
        if os.path.exists('Train_Spectogram_Images'):
            files = os.listdir('Train_Spectogram_Images')
            print(f"Train_Spectogram_Images directory exists with {len(files)} files")
            if len(files) > 0:
                print(f"Skipping spectrogram creation as directory already exists")
                return
                
        # Get Genres and Track IDs from the tracks.csv file
        filename_metadata = "data/tracks_small.csv"
        try:
            tracks = pd.read_csv(filename_metadata, header=2, low_memory=False)
            print("Tracks shape: ", tracks.shape)
            
            # Debug: Print the first few rows and data types
            print("First few rows:")
            print(tracks.head())
            print("Data types:")
            print(tracks.dtypes)
            
            # Convert track IDs to strings explicitly for consistent comparison
            tracks_array = tracks.values
            tracks_id_array = tracks_array[:, 0].astype(str)  # First column contains track IDs
            tracks_genre_array = tracks_array[:, 40]  # Column 40 contains genre info
            
            # Create a dictionary for O(1) lookups
            tracks_id_dict = {id: i for i, id in enumerate(tracks_id_array)}
            
            print(f"Created dictionary with {len(tracks_id_dict)} track IDs")
            # Debug: Print a few examples from the dictionary
            sample_keys = list(tracks_id_dict.keys())[:5]
            print(f"Sample track IDs in dictionary: {sample_keys}")
            
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return

        folder_sample = "Dataset/fma_small"
        if not os.path.exists(folder_sample):
            print(f"ERROR: Dataset directory {folder_sample} does not exist!")
            return
            
        directories = [d for d in os.listdir(folder_sample)
                       if os.path.isdir(os.path.join(folder_sample, d))]
        counter = 0
        if(verbose > 0):
            print("Converting mp3 audio files into mel Spectograms ...")
        if not os.path.exists('Train_Spectogram_Images'):
            os.makedirs('Train_Spectogram_Images')
        for d in directories:
            label_directory = os.path.join(folder_sample, d)
            file_names = [os.path.join(label_directory, f)
                          for f in os.listdir(label_directory)
                          if f.endswith(".mp3")]

            # Convert .mp3 files into mel-Spectograms
            for f in file_names:
                try:
                    track_id = re.search('fma_small/.*/(.+?).mp3', f).group(1)
                    track_id_str = str(track_id)
                    
                    if track_id_str in tracks_id_dict:
                        track_index = tracks_id_dict[track_id_str]
                        genre = tracks_genre_array[track_index]  # Access as 1D array
                        
                        # Check if genre is valid (not '0')
                        if str(genre) != '0':
                            print(f"Processing: {f} - Genre: {genre}")
                            try:
                                y, sr = librosa.load(f)
                                melspectrogram_array = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
                                mel = librosa.power_to_db(melspectrogram_array)
                                # Length and Width of Spectrogram
                                fig_size = plt.rcParams["figure.figsize"]
                                fig_size[0] = float(mel.shape[1]) / float(100)
                                fig_size[1] = float(mel.shape[0]) / float(100)
                                plt.rcParams["figure.figsize"] = fig_size
                                plt.axis('off')
                                plt.axes([0., 0., 1., 1.0], frameon=False, xticks=[], yticks=[])
                                librosa.display.specshow(mel, cmap='gray_r')
                                plt.savefig(f"Train_Spectogram_Images/{counter}_{genre}.jpg", bbox_inches=None, pad_inches=0)
                                plt.close()
                                counter += 1
                            except Exception as e:
                                print(f"Error processing audio file {f}: {e}")
                    else:
                        print(f"Track ID {track_id} not found in metadata, skipping...")
                        
                except Exception as e:
                    print(f"Error extracting track ID from {f}: {e}")
                    
        print(f"Successfully created {counter} spectrograms")
        return

    elif mode == "Test":
        if os.path.exists('Test_Spectogram_Images'):
            return

        folder_sample = "Dataset/DLMusicTest_30"
        counter = 0
        if(verbose > 0):
            print("Converting mp3 audio files into mel Spectograms ...")
        if not os.path.exists('Test_Sepctogram_Images'):
            os.makedirs('Test_Spectogram_Images')
        file_names = [os.path.join(folder_sample, f) for f in os.listdir(folder_sample)
                       if f.endswith(".mp3")]
        # Convert .mp3 files into mel-Spectograms
        for f in file_names:
            test_id = re.search('Dataset/DLMusicTest_30/(.+?).mp3', f).group(1)

            y, sr = librosa.load(f)
            melspectrogram_array = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
            mel = librosa.power_to_db(melspectrogram_array)
            # Length and Width of Spectogram
            fig_size = plt.rcParams["figure.figsize"]
            fig_size[0] = float(mel.shape[1]) / float(100)
            fig_size[1] = float(mel.shape[0]) / float(100)
            plt.rcParams["figure.figsize"] = fig_size
            plt.axis('off')
            plt.axes([0., 0., 1., 1.0], frameon=False, xticks=[], yticks=[])
            librosa.display.specshow(mel, cmap='gray_r')
            plt.savefig("Test_Spectogram_Images/"+test_id+".jpg", cmap='gray_r', bbox_inches=None, pad_inches=0)
            plt.close()
        return

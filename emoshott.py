from moviepy.editor import *
from keras.models import model_from_json
from natsort import natsorted
from datetime import datetime, timedelta
import spacy
import pandas as pd
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
import matplotlib.pyplot as plt
from pandas.plotting import table
import numpy as np
import whisper
from PIL import Image
import cv2
import csv
import os

#extract audio
video = VideoFileClip("./input/input.mp4")
audio = video.audio
audio.write_audiofile("./input/audio.mp3", codec='mp3')

#Generate frames
print("Generating apex frames \n Please wait....")
# Load the Haar cascade file
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Read the video file
video = cv2.VideoCapture('./input/input.mp4')

# Get the frames per second (fps) of the video
fps = video.get(cv2.CAP_PROP_FPS)

# Create a directory named 'frames' to save the frames
if not os.path.exists('frames'):
    os.makedirs('frames')

# Initialize frame count and second count
frame_count = 0
second_count = 1

# Initialize total frame count
total_frame_count = 0

# Initialize variables to store the best frame and its score
best_frame = None
best_score = 0

# Iterate over video frames
while(video.isOpened()):
    ret, frame = video.read()
    if ret:
        total_frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # If this frame has more faces than the current best frame, update the best frame and its score
        if len(faces) > best_score:
            best_frame = frame
            best_score = len(faces)
        # If we have processed enough frames for one second
        if total_frame_count % round(fps) == 0:
            # Calculate the timestamp in seconds
            timestamp = total_frame_count / fps
            # Save the best frame of the last second as a jpg file with the timestamp as its name
            cv2.imwrite(f'frames/{timestamp:.2f}.jpg', best_frame)
            # Reset the best frame and its score
            best_frame = None
            best_score = 0
            # Increase the second count
            second_count += 1
    else:
        break

# Release the video capture
video.release()

# Print a success message
print(f"{second_count - 1} frames were successfully saved in the 'frames' directory.")

#generate subtitle
print("Generating subtitle file...")
# Load the model
model = whisper.load_model("base")

# Transcribe the audio file
result = model.transcribe("./input/audio.mp3")

# Extract the segments with timestamps
segments = result["segments"]

# Generate subtitles in SRT format
with open("./input/subtitles.srt", "w") as file:
    for i, segment in enumerate(segments, start=1):
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]

        # Convert timestamps to SRT format
        start_srt = f"{int(start // 3600):02}:{int((start % 3600) // 60):02}:{int(start % 60):02},{int((start % 1) * 1000):03}"
        end_srt = f"{int(end // 3600):02}:{int((end % 3600) // 60):02}:{int(end % 60):02},{int((end % 1) * 1000):03}"

        # Write the subtitle segment to the file
        file.write(f"{i}\n{start_srt} --> {end_srt}\n{text}\n\n")

print("Subtitles have been generated and saved to subtitles.srt.")

#Detect emotions from generated frames using emotion detection model
print("Identifying emotions...")
# Load the model
json_file = open("./model/emdc.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("./model/emdc.h5")

# Load the face cascade
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Define the labels
labels = {0 : 'angry', 1 : 'contempt', 2 : 'disgust', 3 : 'fear', 4 : 'happy', 5 : 'neutral', 6 : 'sad', 7 : 'surprise'}

# Initialize a dictionary to hold the emotions and corresponding image names
emotion_dict = {emotion: [] for emotion in labels.values()}

# Function to extract features from an image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

# Path to the directory with the images
image_dir = "./frames"

# Get a list of all images in the directory
image_names = os.listdir(image_dir)

# Sort the image names in ascending order
image_names = natsorted(image_names)

# Iterate over all images in the directory
for image_name in image_names:
    # Read the image
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # Extract the face
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))

        # Predict the emotion
        img = extract_features(face)
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]

        # Add the timestamp (extracted from the image name) to the corresponding emotion in the dictionary
        timestamp = float(image_name.split('.')[0])
        emotion_dict[prediction_label].append(timestamp)

# Get the maximum number of timestamps for any emotion
max_timestamps = max(len(timestamps) for timestamps in emotion_dict.values())

# Open the CSV file
with open('./output/emotion_timestamp.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    # Write the header
    writer.writerow(['Emotion'] + [f'Timestamp {i+1}' for i in range(max_timestamps)])

    # Write the rows
    for emotion, timestamps in emotion_dict.items():
        # If this emotion has fewer timestamps than the maximum, fill in the missing fields with empty values
        if len(timestamps) < max_timestamps:
            timestamps += [''] * (max_timestamps - len(timestamps))

        # Write the row
        writer.writerow([emotion] + timestamps)

# Plot
# Read the CSV file
data = pd.read_csv('./output/emotion_timestamp.csv')

# Convert the data to a suitable format
emotions = data['Emotion']
emotion_data = data.drop('Emotion', axis=1)

# Create a new figure
plt.figure(figsize=(10,6))

# Define markers for each emotion
markers = ['o', 's', '^', 'D', '*', 'x', '+', 'v']

# Plot the data
for i, emotion in enumerate(emotions):
    timestamps = emotion_data.iloc[i].dropna()
    if not timestamps.empty:
        plt.scatter(timestamps, [emotion]*len(timestamps), label=emotion, marker=markers[i % len(markers)])

# Add labels and title
plt.xlabel('Timestamp')
plt.ylabel('Emotion')
plt.title('Emotion vs Timestamp')

# Add a legend
plt.legend()

# Save the plot as a PNG file
plt.savefig('./output/emotion_vs_timestamp.png')

print("emotion timestamp log successfully generated")

#NLP and Mapping
print("Correlating Intermediaries....")
nltk.download('sentiwordnet')
nltk.download('wordnet')

# Define the time format
FMT = '%H:%M:%S,%f'

# Load the English language model
nlp = spacy.load("en_core_web_sm")

def extract_words_with_actions(subtitle_file, start_time, end_time):
    # Convert timestamp strings to timedelta objects for comparison
    FMT = '%H:%M:%S,%f'
    start_time = datetime.strptime(start_time, FMT) - datetime.strptime("00:00:00,000", FMT)
    end_time = datetime.strptime(end_time, FMT) - datetime.strptime("00:00:00,000", FMT)

    action_words = []
    with open(subtitle_file, 'r') as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        # Skip sequence number
        i += 1

        # Parse start and end times
        time_line = lines[i].strip()
        if ' --> ' in time_line:
            timestamp_start_str, timestamp_end_str = time_line.split(' --> ')
            timestamp_start = datetime.strptime(timestamp_start_str, FMT) - datetime.strptime("00:00:00,000", FMT)
            timestamp_end = datetime.strptime(timestamp_end_str, FMT) - datetime.strptime("00:00:00,000", FMT)
            i += 1

            # Check if the subtitle entry falls within the specified time range
            if start_time <= timestamp_start <= end_time or start_time <= timestamp_end <= end_time:
                # Gather all lines of text for this subtitle entry
                text_lines = []
                while i < len(lines) and lines[i].strip() != '':
                    text_lines.append(lines[i].strip())
                    i += 1

                 # Process the text using SpaCy
                doc = nlp(' '.join(text_lines))
                # Extract verbs, nouns, 
                for token in doc:
                    if token.pos_ in ['VERB', 'PROPN', 'ADJ', 'ADV']:  # Add other POS tags as needed
                        # Perform sentiment analysis on the token
                        synsets = list(swn.senti_synsets(token.text))
                        if synsets:
                            # Get the first synset (most common sense)
                            synset = synsets[0]
                            # If the word has a strong positive or negative sentiment, add it to the list
                            if synset.pos_score() > 0.5 or synset.neg_score() > 0.2:
                                action_words.append(token.text)
                # Extract named entities
                for ent in doc.ents:
                    action_words.append(ent.text)
                # Convert action_words to a set to remove duplicates
                action_words = set(action_words)
                # Convert action_words back to a list
                action_words = list(action_words)
            else:
                # Skip lines of text for this subtitle entry
                while i < len(lines) and lines[i].strip() != '':
                    i += 1
        else:
            # If the line doesn't contain ' --> ', skip it
            i += 1

        # Skip blank line
        i += 1

    return action_words

# Load the emotion-image CSV
emotion_image_df = pd.read_csv('./output/emotion_timestamp.csv', index_col=0)

# Create a new DataFrame to store the results
result_df = pd.DataFrame(columns=['Emotion', 'Action Words'])

# Iterate over the rows of the emotion-image DataFrame
for emotion, row in emotion_image_df.iterrows():
    # Iterate over the images for this emotion
    images = row.dropna().tolist()
    if images:
        # Extract the timestamps from the image file names
        timestamps = [float(image_file) for image_file in images]

        # Use the first timestamp as the start time and the last timestamp as the end time
        start_time = str(timedelta(seconds=int(min(timestamps))))
        end_time = str(timedelta(seconds=int(max(timestamps) + 1)))

        # Convert the start and end times to the required format
        start_time = datetime.strftime(datetime.strptime(start_time, '%H:%M:%S'), FMT)
        end_time = datetime.strftime(datetime.strptime(end_time, '%H:%M:%S'), FMT)

        # Call the function with appropriate parameters
        subtitle_file = './input/subtitles.srt'  # Enter the path to your subtitle file
        action_words = extract_words_with_actions(subtitle_file, start_time, end_time)

        # Add the results to the new DataFrame
        new_row = pd.DataFrame({'Emotion': [emotion], 'Action Words': [', '.join(action_words)]})
        result_df = pd.concat([result_df, new_row], ignore_index=True)

# Save the new DataFrame to a CSV file
result_df.to_csv('./output/emotion_action_words.csv', index=False)

# Tabulate
# Path to the CSV file
csv_file_path = './output/emotion_action_words.csv'

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Create a new figure
fig, ax = plt.subplots(figsize=(12, 2)) # Adjust size as needed
ax.axis('tight')
ax.axis('off')

# Create the table and adjust the cell size
tab = table(ax, df, loc='center', cellLoc='center', colWidths=[0.2]*len(df.columns))

# Auto-size the columns to fit the content
tab.auto_set_column_width(col=list(range(len(df.columns))))

# Save the plot as a PNG file
plt.savefig('./output/emotion_action_words_table.png', bbox_inches='tight', pad_inches=0.05)

print("Map generate successfully")

print("Opening results...")
img1 = Image.open("./output/emotion_vs_timestamp.png")
img2 = Image.open("./output/emotion_action_words_table.png")
img1.show()
img2.show()

# EMO-SHOTT : AI Emotional Trigger Detector


Emotional well-being is crucial for a healthy life, impacting physical health, mental health, and relationships. Understanding and managing emotions can enhance overall well-being. Our project aims to bridge the gap in current systems by detecting emotional triggers through multimodal analysis of facial expressions and speech.

## Key Features

- **Emotion Trigger Detection:** Combines facial motion analysis and speech-to-text processing.
- **Enhanced Emotion Classification:** Expands recognition from 3 to 8 emotions – Anger, Disgust, Fear, Happiness, Sadness, Surprise, Neutral, and Contempt.
- **Data Processing:**
  - **FER-2013 PLUS Dataset:** 48x48 pixel grayscale face images with improved labeling.
  - **OpenAI Whisper:** Speech recognition and translation for audio transcriptions.
  - **spaCy NLP:** Part-of-Speech tagging and Named Entity Recognition (NER) for trigger identification.

## Methodology

![Picture1](https://github.com/user-attachments/assets/0493d986-29c1-4710-a299-be95e8000946)


### Facial Analysis

- Feature Extraction using Haar features.
- AdaBoost for feature selection and classification.
- Cascade Classifiers for improved efficiency.

### Speech Processing

- Audio split into 30-sec chunks, processed via Whisper encoder-decoder.
- NLP mapping of triggers using spaCy.

### Model Training

- Tested various architectures including VGG16, TensorFlow Keras, PyTorch DDAMNET, and CNN.
- Achieved validation accuracy of 86.36% after 100 epochs.
![Picture2](https://github.com/user-attachments/assets/d050d735-045f-43f7-8215-d4e982f85ffb)
![Picture3](https://github.com/user-attachments/assets/1de17751-fbe2-49a8-975c-9f8c00a6dc93)
![Picture4](https://github.com/user-attachments/assets/f882a944-0a5c-4a9f-8d00-c0637ece3369)


## Achievements

- Successfully mapped emotions to triggers by correlating subtitles and apex frames.
- Developed a robust system enhancing mental well-being through accurate emotion detection.


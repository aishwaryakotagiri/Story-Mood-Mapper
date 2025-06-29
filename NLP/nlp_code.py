import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt

# 1. Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# 2. Load the NRC Emotion Lexicon CSV
df = pd.read_csv("NRC-Emotion-Lexicon.csv")

# 3. Keep only English words and emotion columns
emotion_columns = ['Positive', 'Negative', 'Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']
df = df[['English (en)'] + emotion_columns]
df = df.rename(columns={'English (en)': 'word'})
df['word'] = df['word'].str.lower()

# 4. Convert it to a dictionary: word → emotion scores
emotion_dict = df.set_index('word').T.to_dict()

# 5. Text preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    cleaned = [word for word in tokens if word.isalpha() and word not in stop_words]
    return cleaned

# 6. Emotion detection function
def detect_emotions(text):
    words = preprocess_text(text)
    emotion_score = {emotion: 0 for emotion in emotion_columns}

    for word in words:
        if word in emotion_dict:
            for emotion in emotion_columns:
                emotion_score[emotion] += emotion_dict[word].get(emotion, 0)

    return emotion_score

# 7. Visualization function
def visualize_emotions(emotion_score):
    emotions = list(emotion_score.keys())
    scores = list(emotion_score.values())

    plt.figure(figsize=(10, 6))
    plt.bar(emotions, scores, color='skyblue')
    plt.title("Emotion Distribution in Text", fontsize=16)
    plt.xlabel("Emotions", fontsize=12)
    plt.ylabel("Intensity", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 8. Example usage
sample_text = input("Enter the text for emotion analysis: ")

# Detect emotions in the text
emotions = detect_emotions(sample_text)

# Display the detected emotions
print("\n🔍 Detected Emotions:")
for emotion, score in emotions.items():
    print(f" - {emotion}: {score}")

# Visualize the detected emotions
visualize_emotions(emotions)

STORY MOOD MAPPER

Hey there! 
Have you ever written something and wondered — “What does this feel like?”
That’s exactly what Story Mood Mapper helps you discover.

This little tool reads your words and gently tells you what emotions live inside them. Whether you’re writing a story, journaling your thoughts, or just exploring language, it shows you the emotional shades of your text — joy, sadness, fear, trust, and more.

What it does
- Reads any English text you give it

- Cleans it up using NLTK (no stopwords, no noise)

- Looks up each word’s emotions from the NRC Emotion Lexicon

- Shows you a pretty bar chart of how much of each emotion your story carries

Whether you’re writing a novel or just curious about how your words “feel” — this tool brings those vibes to life!

🛠️What's inside?
story-mood-mapper/
├── nlp_code.py                 → The emotion detection script
├── NRC-Emotion-Lexicon.csv     → The emotion dictionary
└── README.md                   → You’re reading this!

How to use it
Clone the project:
git clone https://github.com/aishwaryakotagiri/story-mood-mapper.git
cd story-mood-mapper

Install the libraries:
pip install nltk pandas matplotlib

Run the script:
python nlp_code.py

Type or paste your story snippet:
Enter the text for emotion analysis: 
The sky was quiet, and her heart even quieter.
Get a beautiful breakdown of emotions — both in text and a chart!

Emotions it detects:
-Joy 
-Sadness 
-Fear 
-Anger 
-Trust 
-Surprise 
-Anticipation 
-Disgust 
-Positive 
-Negative 

Why I made this
I’m a writer, a dreamer, and a tech enthusiast.
And sometimes, I just want to know — what is the emotional pulse of what I wrote?

So I made this as a fun blend of AI and storytelling — a tool for writers who care about how their words feel. It helps me reflect on my mood, my story’s energy, and even how a reader might experience it.

Maybe it’ll do the same for you 

About Me:
I'm Aishwarya Kotagiri, a student building things with machine learning and heart.

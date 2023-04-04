import json

import nltk
import pyaudio
import pymorphy2

from vosk import Model, KaldiRecognizer

from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

nltk.download('popular')
nltk.download('stopwords')

from nltk.corpus import stopwords



# variables for tokenized text
filtered_text=[]

# variables for stemming
morph = pymorphy2.MorphAnalyzer()

model = Model('ru-model')
rec = KaldiRecognizer(model, 16000)
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)

stream.start_stream()

def listen():
    while True:
        data = stream.read(4000, exception_on_overflow=False)
        if rec.AcceptWaveform(data) and len(data) > 0:
            answer = json.loads(rec.Result())

            if answer['text']:
                yield answer['text']

for text in listen():
    filtered_text.clear()
    for token in word_tokenize(text, language="russian"):
        if token not in stopwords.words('russian'):
            filtered_text.append(morph.parse(token)[0].normal_form)

    print(filtered_text)
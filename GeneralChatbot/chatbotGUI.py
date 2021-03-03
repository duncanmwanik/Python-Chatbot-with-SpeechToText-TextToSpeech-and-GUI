import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import pyttsx3
import speech_recognition as sr
import pyaudio
from keras.models import load_model
import json
import random

lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


def speak(phrase):
    engine = pyttsx3.init()  # object creation

    """ RATE"""
    rate = engine.getProperty('rate')  # getting details of current speaking rate
    print("--------")  # printing current voice rate
    engine.setProperty('rate', 150)  # setting up new voice rate

    """VOLUME"""
    volume = engine.getProperty('volume')  # getting to know current volume level (min=0 and max=1)
    print("--------------")  # printing current volume level
    engine.setProperty('volume', 1.0)  # setting up volume level  between 0 and 1

    """VOICE"""
    voices = engine.getProperty('voices')  # getting details of current voice
    engine.setProperty('voice', voices[1].id)  # changing index, changes voices. 1 for female, 0 for female

    engine.say(phrase)
    engine.runAndWait()
    engine.stop()


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = get_response(ints, intents)
    return res


def get_speech_to_text():
    r = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        return text
    except:
        return 'How old are you'


# Creating GUI with tkinter
from tkinter import *

g = Tk()
EntryBox = Text(g, bd=0, bg="white", width="40", height="10", font="Arial")


def get_text():
    # Uses text from the text entry field
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
        speak(res)

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


def get_speech():
    msg = get_speech_to_text()

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
        speak(res)  # Speak the chatbot response using Text-To-Speech

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


# Building the user interface
g.title("MyChatbot")
g.geometry("400x500")
g.resizable(width=FALSE, height=FALSE)

# Create Chat window
ChatLog = Text(g, bd=0, bg="white", height="8", width="50", font="Arial", )
ChatLog.config(state=DISABLED)

# Bind scrollbar to Chat window
scrollbar = Scrollbar(g, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

# Create Button to speak message
SayButton = Button(g, font=("Verdana", 12, 'bold'), text="Speak", width="12", height=5,
                   bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff', command=get_speech)

# Create Button to send message
SendButton = Button(g, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff', command=get_text)

# Create the box to enter message
EntryBox = Text(g, bd=0, bg="white", width="40", height="10", font="Arial")
EntryBox.bind("<Return>", (lambda event: get_text()))

# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SayButton.place(x=6, y=401, height=40)
SendButton.place(x=6, y=450, height=40)
g.mainloop()

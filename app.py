import streamlit as st
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import pickle

stemmer = PorterStemmer()

def text_transformation(text):
  text = text.lower()
  text = nltk.word_tokenize(text)
  new_text = []

  for i in text:
    if i.isalnum():
      new_text.append(i)

  text = new_text[:]
  new_text.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      new_text.append(i)

  text = new_text[:]
  new_text.clear()

  for i in text:
    new_text.append(stemmer.stem(i))

  return ' '.join(new_text)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email/SMS Spam Classifier')
user_input = st.text_input('Enter your message: ')

if st.button('Predict'):
  transformed_input = text_transformation(user_input)
  input_vector = tfidf.transform([transformed_input])
  result = model.predict(input_vector)[0]

  if result == 1:
      st.header('The message is "Spam"')
  else:
      st.header('The message is "Not spam"')
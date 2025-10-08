import pickle
import string
import streamlit as st
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import nltk 

nltk.download('punkt_tab')
nltk.download('stopwords')

# ["running", "runs", "runner" --> run]

pr = PorterStemmer()


# function to transform text
def transform_text(text : str):


    # convert into lower case
    text = text.lower()

    # convert into tokens
    text = word_tokenize(text)

    # remove special character

    list1 = []
    
    for i in text:

        if(i.isalnum()):

            list1.append(i) # append into list the alpha num chracter


    #  Removing stop words and punctation
    text = list1[:]
    list1.clear()

    for i in text:

        if i not in stopwords.words('english') and i not in string.punctuation:

            list1.append(i)

    # Stemming

    text = list1[:]
    list1.clear()

    for i in text:

        list1.append(pr.stem(i))



    return ' '.join(list1)

# load the model and vectorizer

model = pickle.load(open('model.pkl' , 'rb'))
tfidf = pickle.load(open('vectorizer.pkl' , 'rb'))

# Add the title
st.title('Email/SMS spam classifier')


# take input form the user
input_text = st.text_area('Enter the message ')


# add the predict button
if st.button('Predict'):

    # 1. preprocess

    transformed_sms = transform_text(input_text)

    # 2. vectorize

    vector_input = tfidf.transform([transformed_sms])

    # 3. predict

    result = model.predict(vector_input)[0]

    # 4. Display

    if(result == 1):
        st.header('Spam')

    else:

        st.header('Not spam')

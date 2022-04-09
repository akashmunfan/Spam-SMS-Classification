from flask import *  
import joblib
import pickle

# Importing essential libraries for performing NLP
import nltk
import re
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
# Importing essential libraries
import numpy as np
import pandas as pd

app = Flask(__name__) #creating the Flask class object   
app.secret_key = b'Akash@2867/'
model = joblib.load('model.joblib')

@app.route('/') #decorator drfines the   
def index():  
    return  render_template('index.html'); 

@app.route('/predict',methods=['POST'])  
def predict():
    status = ""
    sms=request.form['sms']
    output = predict_spam(sms)
    print(output)
    if output:
        status = "Spam"
    else:
        status = "Ham(Normal Message)"
    return render_template('index.html', status=status)
def predict_spam(sample_message):
    wnl = WordNetLemmatizer()


    tfidf = pickle.load( open( "tfidf.pickle", "rb") )

    sample_message = re.sub(pattern='[^a-zA-Z]',repl=' ', string = sample_message)
    sample_message = sample_message.lower()
    sample_message_words = sample_message.split()
    sample_message_words = [word for word in sample_message_words if not word in set(stopwords.words('english'))]
    final_message = [wnl.lemmatize(word) for word in sample_message_words]
    final_message = ' '.join(final_message)

    temp = tfidf.transform([final_message]).toarray()
    return model.predict(temp)
if __name__ =='__main__':  
    app.run(debug = True)  
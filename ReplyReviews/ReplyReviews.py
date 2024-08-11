from flask import Blueprint, render_template, request, url_for, redirect
import json
#imports for model
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ReplyReviews = Blueprint('ReplyReviews', __name__, template_folder='templates')
#globle varable
DF = pd.DataFrame()

@ReplyReviews.route('/', methods=['GET', 'POST'])
def index():
   if request.method == 'POST':
      global DF
      # check which form was submitted
      if 'upload' in request.form:
         print(request.method, request.files)
         #train dataset
         DF = pd.read_csv(request.files.get('file'))
         columns = list(DF.columns)
         return render_template('index.html' , columns = columns)
      elif 'submit' in request.form:
         print(request.method, request.form)
         # Define the TF-IDF vectorizer
         tfidf = TfidfVectorizer(lowercase=True)

         #final Model Function
         def Review_Response(df: pd.DataFrame, text: str, reply: str, new_input: str):
            # Fit and transform the texts into TF-IDF vectors
            texts_tfidf = tfidf.fit_transform(df[text].astype(str))
            # Get the similarity scores between the input text and the training texts
            def get_similarity_scores(input_text):
               input_tfidf = tfidf.transform([input_text])
               similarity_scores = cosine_similarity(texts_tfidf, input_tfidf)
               return similarity_scores
            # Get the index of the most similar training text
            def get_most_similar_index(input_text):
                  similarity_scores = get_similarity_scores(input_text)
                  most_similar_index = similarity_scores.argmax()
                  return most_similar_index
            # Train the model by getting the index of the most similar training text for each input text
            def input_text(input_text):
               most_similar_index = get_most_similar_index(input_text)
               response = df.iloc[most_similar_index][reply]
               return response
            return input_text(new_input)
         
         #arguments for the model  
         new_input = request.form.get('input')
         text_col=request.form.get('x_col')
         reply_col=request.form.get('y_col')
         #function Call
         results = Review_Response(DF, text_col, reply_col, new_input)
         return render_template('index.html' , dataToRender = results)
   else:
      return render_template('index.html')	

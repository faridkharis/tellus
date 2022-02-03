from flask import Flask, render_template, url_for, redirect, request, flash, session
import pandas as pd
import numpy as np
from google_play_scraper import app, Sort, reviews_all
import matplotlib.pyplot as plt
import re
import nltk
from nltk import word_tokenize
import sys
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans


app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'

class DataStore():
    df = None
    clean_data2 = None

data = DataStore()


@app.route("/", methods=["POST", "GET"])
def index():
    if "username" in session:
        return redirect(url_for("home"))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin':
            session['username'] = username
            return redirect(url_for("home"))
        else:
            return render_template("index.html")
    return render_template("index.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/scraping", methods=["POST", "GET"])
def scraping():
    if "username" in session:
        if request.method == 'POST':
            id_url = request.form['id_url']

            result = reviews_all(
                app_id = id_url,
                sleep_milliseconds=0,
                lang='id',
                country='id', 
                sort=Sort.MOST_RELEVANT,
                count=50,
            )

            df = pd.DataFrame(result)
            df = df[['userName', 'at', 'content', 'score', 'reviewCreatedVersion']]

            data.df = df
            # session["df"] = df

            return render_template("review.html", tables=[df.to_html(classes='empTable display dataTable table-review')], titles=df.columns.values)
        else:
            return render_template("scraping.html") 
    else:
        return redirect(url_for("index"))


@app.route("/review")
def review():
    if "username" in session:
        return render_template("review.html")
    else:
        return redirect(url_for("index"))

@app.route("/stopword")
def stopword():
    if "username" in session:
        return render_template("stopword.html")
    else:
        return redirect(url_for("index"))

@app.route("/normalization")
def normalization():
    if "username" in session:
        return render_template("normalization.html")
    else:
        return redirect(url_for("index"))

@app.route("/clean")
def clean():
    if "username" in session:
        df = data.df

        fSlang = 'static/files/slang_word.csv'
        bahasa = 'id'
        sw = open(fSlang,encoding='utf-8', errors='ignore', mode='r'); SlangS=sw.readlines(); sw.close()
        SlangS = {slang.strip().split(',')[0]:slang.strip().split(',')[1] for slang in SlangS}

        Docs = []
        st = open('static/files/stop_word.txt', "r", encoding="utf-8", errors='replace')
        Docs.append(st.readlines()); st.close()
        stops = set([t.strip() for t in Docs[0]])
        for x in ['aplikasi', 'kulina', 'makan', 'makanan', 'menu']:
            stops.add(x)

        def formaldanstop(t):
            t = word_tokenize(t)
            for i,x in enumerate(t):
                if x in SlangS.keys():
                    t[i] = SlangS[x]
            return ''.join(' '.join(x for x in t if x not in stops))

        reviews = [x for x in df['content']]

        reviews_preprocessing1 = []
        for x in reviews:
            # Remove URLs
            pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
            x = re.sub(pattern,' ',x)
            # Convert to lowercase
            x = x.lower()
            # Convert www.* or https?://*
            x = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',x)
            # Remove symbols
            x = re.sub(r'[^.,a-zA-Z0-9 \n\.]',' ',x)
            x = x.replace(',',' ').replace('.',' ')
            # Remove additional whitespace
            x = re.sub('[\s]+',' ',x)
            # trim
            x = x.strip('\'"')
            reviews_preprocessing1.append(str(x))

        reviews_preprocessing2 = list(map(formaldanstop,reviews_preprocessing1))

        stopword = StopWordRemoverFactory().create_stop_word_remover()
        stemmer = StemmerFactory().create_stemmer()

        clean_reviews = []

        for i, term in enumerate(reviews_preprocessing2):
            stop = stopword.remove(term)
            stem = stemmer.stem(stop)
            print('kalimat ke:', i+1, 'dari', len(reviews_preprocessing2))
            clean_reviews.append(stem)
        
        clean_data1 = pd.DataFrame(clean_reviews, columns=['clean_text'])
        # clean_data1.head()

        clean_data1['userName'] = df['userName']
        clean_data1['at'] = df['at']
        clean_data1['score'] = df['score']
        clean_data1['reviewCreatedVersion'] = df['reviewCreatedVersion']
        
        clean_data2 = clean_data1.dropna()
        print(clean_data2.shape)
        # clean_data2.head()


        return render_template("clean.html", tables=[clean_data2.to_html(classes='empTable display dataTable table-review')], titles=clean_data2.columns.values)
        # return render_template("clean.html")

    else:
        return redirect(url_for("index"))

@app.route("/clustering")
def clustering():
    if "username" in session:

        clustering_data = data.clean_data2

        vectorizer = CountVectorizer()
        tfidf_transformer = TfidfTransformer()

        vector_data = vectorizer.fit_transform(clustering_data['clean_text'])
        tfidf_data = tfidf_transformer.fit_transform(vector_data)
        # vector_data.shape

        kmeans = KMeans(n_clusters=4, random_state=0).fit(tfidf_data)
        result = kmeans.labels_
        # len(result)

        clustering_data['Cluster'] = result
        # clean_data2.head()


        return render_template("clustering.html", tables=[clustering_data.to_html(classes='empTable display dataTable table-review')], titles=clustering_data.columns.values)
    else:
        return redirect(url_for("index"))

@app.route("/logout")
def logout():
    if "username" in session:
        session.pop("username")
        return redirect(url_for("index"))
    else:
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
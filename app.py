from flask import Flask, make_response, render_template, url_for, redirect, request, flash, session, send_file
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
from wordcloud import WordCloud, STOPWORDS


app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'

class DataStore():
    df = None
    clean_data2 = None
    clustering_data = None
    stops = None
    visualization_title = None
    visualization_id = None

data = DataStore()


@app.route("/", methods=["POST", "GET"])
def index():
    if "username" in session:
        return redirect(url_for("home"))
    # ================ LOGIN VALIDATION ================
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
        # ================ SCRAPING DATA ================
        if request.method == 'POST':
            id_url = request.form['id_url']

            result = reviews_all(
                app_id=id_url,
                sleep_milliseconds=0,
                lang='id',
                country='id', 
                sort=Sort.MOST_RELEVANT,
            )

            data.visualization_id = id_url

            df = pd.DataFrame(result)

            hist = pd.value_counts(df['score'])
            fig = hist.plot(kind='bar', figsize=(4, 4)).get_figure()
            fig.savefig('static/files/score_pie.jpg')
            
            df['Nama Pengguna'] = df['userName']
            df['Waktu'] = df['at']
            df['Ulasan'] = df['content']
            df = df[['Nama Pengguna', 'Waktu', 'Ulasan']]
            
            data.df = df

            return render_template("review.html", tables=[df.to_html(classes='empTable display dataTable table-review')], titles=['na'])
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

@app.route("/download")
def download():
    if "username" in session:
        df = data.df
        resp = make_response(df.to_csv())
        resp.headers["Content-Disposition"] = "attachment; filename=data.csv"
        resp.headers["Content-Type"] = "text/csv"
        return resp
    else:
        return redirect(url_for("index"))

@app.route("/stopword")
def stopword():
    if "username" in session:
        df_stopword = pd.read_csv('static/files/stop_word.txt')
        
        df_stopword['Stopword'] = df_stopword['ada']
        df_stopword = df_stopword[['Stopword']]
        
        return render_template("stopword.html", tables=[df_stopword.to_html(classes='empTable display dataTable table-review')], titles=[''])
    else:
        return redirect(url_for("index"))

@app.route("/normalization")
def normalization():
    if "username" in session:
        df_normalization = pd.read_csv('static/files/slang_word.csv')
        
        df_normalization['Kata Slang'] = df_normalization['aj']
        df_normalization['Normalisasi'] = df_normalization['saja']
        df_normalization = df_normalization[['Kata Slang', 'Normalisasi']]
        
        return render_template("normalization.html", tables=[df_normalization.to_html(classes='empTable display dataTable table-review')], titles=[''])
    else:
        return redirect(url_for("index"))

@app.route("/clean")
def clean():
    if "username" in session:
        df = data.df
        # ================ NORMALIZATION ================
        fSlang = 'static/files/slang_word.csv'
        # bahasa = 'id'
        sw = open(fSlang,encoding='utf-8', errors='ignore', mode='r'); SlangS=sw.readlines(); sw.close()
        SlangS = {slang.strip().split(',')[0]:slang.strip().split(',')[1] for slang in SlangS}

        # ================ STOPWORD REMOVAL ================
        Docs = []
        st = open('static/files/stop_word.txt', "r", encoding="utf-8", errors='replace')
        Docs.append(st.readlines()); st.close()
        stops = set([t.strip() for t in Docs[0]])
        for x in ['aplikasi', 'kulina', 'makan', 'makanan', 'menu']:
            stops.add(x)
        data.stops = stops

        # ================ TOKENIZATION ================
        def formaldanstop(t):
            t = word_tokenize(t)
            for i,x in enumerate(t):
                if x in SlangS.keys():
                    t[i] = SlangS[x]
            return ''.join(' '.join(x for x in t if x not in stops))

        reviews = [x for x in df['Ulasan']]

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

        # ================ STEMMING ================
        stopword = StopWordRemoverFactory().create_stop_word_remover()
        stemmer = StemmerFactory().create_stemmer()

        clean_reviews = []

        for i, term in enumerate(reviews_preprocessing2):
            stop = stopword.remove(term)
            stem = stemmer.stem(stop)
            clean_reviews.append(stem)
        
        clean_data1 = pd.DataFrame(clean_reviews, columns=['Ulasan Bersih'])

        clean_data1['Nama Pengguna'] = df['Nama Pengguna']
        clean_data1['Waktu'] = df['Waktu']
        
        clean_data2 = clean_data1.dropna()

        data.clean_data2 = clean_data2

        return render_template("clean.html", tables=[clean_data2.to_html(classes='empTable display dataTable table-review')], titles=['na'])
    else:
        return redirect(url_for("index"))

@app.route("/clustering")
def clustering():
    if "username" in session:

        clustering_data = data.clean_data2

        # ================ TF-IDF ================
        vectorizer = CountVectorizer()
        tfidf_transformer = TfidfTransformer()

        vector_data = vectorizer.fit_transform(clustering_data['Ulasan Bersih'])
        tfidf_data = tfidf_transformer.fit_transform(vector_data)

        # ================ K-MEANS CLUSTERING ================
        kmeans = KMeans(n_clusters=4, random_state=0).fit(tfidf_data)
        kmeans_result = kmeans.labels_

        clustering_data['Klaster'] = kmeans_result

        data.clustering_data = clustering_data

        return render_template("clustering.html", tables=[clustering_data.to_html(classes='empTable display dataTable table-review')], titles=['na'])
    else:
        return redirect(url_for("index"))

@app.route("/visualization")
def visualization():
    if "username" in session:
        visualization_data = data.clustering_data
        stops = data.stops
        visualization_id = data.visualization_id

        hist_2 = pd.value_counts(visualization_data['Klaster'])
        fig_2 = hist_2.plot(kind='pie').get_figure()
        fig_2.savefig('static/files/cluster_pie.jpg')

        # ================ WORDCLOUD ================
        cluster_0 = visualization_data[visualization_data.Klaster==0]
        cluster_1 = visualization_data[visualization_data.Klaster==1]
        cluster_2 = visualization_data[visualization_data.Klaster==2]
        cluster_3 = visualization_data[visualization_data.Klaster==3]

        comment_words = ''
        stopwords = set(stops)

        for val in cluster_0['Ulasan Bersih']:
            # typcaste each val to string
            val = str(val)
            # split the value
            tokens = val.split()
            # covert each token into lowercase
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()
            comment_words += " ".join(tokens)+" "
        
        wordcloud = WordCloud(width = 400, height = 400,
                            background_color = 'white', stopwords = stopwords,
                            min_font_size = 10).generate(comment_words)
        # plot the WordCloud image
        plt.figure(figsize = (4, 4), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.tight_layout(pad = 0)
        plt.savefig('static/files/wordcloud_0.jpg')


        comment_words = ''
        stopwords = set(stops)

        for val in cluster_1['Ulasan Bersih']:
            # typcaste each val to string
            val = str(val)
            # split the value
            tokens = val.split()
            # covert each token into lowercase
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()
            comment_words += " ".join(tokens)+" "
        
        wordcloud = WordCloud(width = 400, height = 400,
                            background_color = 'white', stopwords = stopwords,
                            min_font_size = 10).generate(comment_words)
        # plot the WordCloud image
        plt.figure(figsize = (4, 4), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.tight_layout(pad = 0)
        plt.savefig('static/files/wordcloud_1.jpg')


        comment_words = ''
        stopwords = set(stops)

        for val in cluster_2['Ulasan Bersih']:
            # typcaste each val to string
            val = str(val)
            # split the value
            tokens = val.split()
            # covert each token into lowercase
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()
            comment_words += " ".join(tokens)+" "
        
        wordcloud = WordCloud(width = 400, height = 400,
                            background_color = 'white', stopwords = stopwords,
                            min_font_size = 10).generate(comment_words)
        # plot the WordCloud image
        plt.figure(figsize = (4, 4), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.tight_layout(pad = 0)
        plt.savefig('static/files/wordcloud_2.jpg')


        comment_words = ''
        stopwords = set(stops)

        for val in cluster_3['Ulasan Bersih']:
            # typcaste each val to string
            val = str(val)
            # split the value
            tokens = val.split()
            # covert each token into lowercase
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()
            comment_words += " ".join(tokens)+" "
        
        wordcloud = WordCloud(width = 400, height = 400,
                            background_color = 'white', stopwords = stopwords,
                            min_font_size = 10).generate(comment_words)
        # plot the WordCloud image
        plt.figure(figsize = (4, 4), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.tight_layout(pad = 0)
        plt.savefig('static/files/wordcloud_3.jpg')

        return render_template("visualization.html", visualization_id=visualization_id)
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
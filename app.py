from flask import Flask, render_template, url_for, redirect, request, flash, session
import os
import csv
import pandas as pd
import numpy as np
from google_play_scraper import app, Sort, reviews_all


app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'


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
            df = df[['userName', 'at', 'content']]

            return render_template("review.html", tables=[df.to_html(classes='data')], titles=df.columns.values)
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
        return render_template("clean.html")
    else:
        return redirect(url_for("index"))

@app.route("/clustering")
def clustering():
    if "username" in session:
        return render_template("clustering.html")
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
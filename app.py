from flask import Flask, render_template, url_for, redirect, request
from flask_mysqldb import MySQL

app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'tutorial_flaskdb'
mysql = MySQL(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/review")
def review():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM computer")
    rv = cur.fetchall()
    cur.close()
    return render_template("review.html", computers=rv)


@app.route('/simpan',methods=["POST"])
def simpan():
    nama = request.form['nama']
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO computer (data) VALUES (%s)",(nama,))
    mysql.connection.commit()
    return redirect(url_for('review'))



@app.route("/stopword")
def stopword():
    return render_template("stopword.html")

@app.route("/normalization")
def normalization():
    return render_template("normalization.html")

@app.route("/clean")
def clean():
    return render_template("clean.html")

@app.route("/clustering")
def clustering():
    return render_template("clustering.html")

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, render_template, session, request, url_for, redirect, Markup

app = Flask(__name__)

# index
@app.route("/",methods=['GET','POST'])
@app.route("/index",methods=['GET','POST'])
@app.route("/home",methods=['GET','POST'])
def index():
    if request.method == 'GET':
        return render_template("index.html")
    else:
        return render_template("index.html")

@app.route("/submit", methods=['GET','POST'])
def submit():
    return render_template("index.html")

if __name__ == "__main__":
    app.debug = True
    app.secret_key = "appapp"
    app.run(host='0.0.0.0',port=8000)
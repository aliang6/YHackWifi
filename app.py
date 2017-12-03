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
        button = request.form['button']
        if button == "Get My Quote":
            userPredictions = grabUserData()
            return render_template("index.html", participants=userPredictions)
        else:
            writeAllData()
            return render_template("index.html", participants=group_participants)

if __name__ == "__main__":
    app.debug = True
    app.secret_key = "appapp"
    app.run(host='0.0.0.0',port=8000)
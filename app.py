from flask import Flask, render_template, session, request, url_for, redirect, Markup
from linearmodel import grabUserData, writeAllData, grabPrediction, setup

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
        print(request.form)
        if button == "Get My Quote":
            userPredictions = grabUserData(request.form)
            return render_template("index.html", participants=userPredictions)
        else:
            writeAllData()
            return render_template("index.html", writeAllData = True)

@app.route("/input",methods=['GET','POST'])
def input():
    if request.method == 'GET':
        return render_template("input.html")
    else:
        input=dict(request.form)
        for i in input:
            input[i]=input[i][0]
        userPrediction = grabPrediction(input)
        return render_template("par_coor.html", plan=userPrediction)


if __name__ == "__main__":
    app.debug = True
    app.secret_key = "appapp"
    app.run(host='0.0.0.0',port=8000)
    setup(group_data, -1, -1, -1)
    
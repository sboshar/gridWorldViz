from flask import Flask, redirect, url_for, render_template, request, session,jsonify
from qlearning import QLearning
app = Flask(__name__)

#var = [[i for i in range(10)] for j in raX nge(10)]
app.secret_key = "secrettunnel"

# @app.route("/admin")
# def admin():
  # return redirect(url_for("home"))

@app.route("/")
def home():
  #check isf any info in session  
  if "epochs" in session:
    epochs = session["epochs"]
    gamma = session["gamma"]
    qagent = QLearning(num_epochs=int(epochs), gamma=float(gamma))
    reward, count = qagent.run(qagent.epsLinear, qagent.epsLinear)
    data = {'reward': reward, 'count': count}
    return render_template("index.html", data=data)
  else:
    return redirect(url_for("getvar"))

@app.route("/var", methods=["POST", "GET"])
def getvar():
  print("here")
  if request.method == "POST":
    epochs = request.form["epochs"]
    gamma = request.form["gamma"]
    session["epochs"] = epochs
    session["gamma"] = gamma
    return redirect(url_for("home"))
  else:
    print("oops")
    return render_template("getvar.html")


if __name__ =="__main__":
  app.run(debug=True)
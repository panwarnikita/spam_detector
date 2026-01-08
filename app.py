from flask import Flask, render_template, request, redirect, url_for
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message'].strip()

    if message == "":
        return redirect(url_for('home'))

    data = vectorizer.transform([message])
    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Spam"
    else:
        result = "Not Spam"

    return redirect(url_for('home', result=result))

if __name__ == "__main__":
    app.run(debug=True)







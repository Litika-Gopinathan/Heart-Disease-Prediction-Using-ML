from flask import Flask ,  render_template , request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl","rb"))

@app.route("/")
def home():
       return render_template("index.html")


@app.route("/predict", methods = ['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    prediction = model.predict(final_features)
    output = round(prediction[0],2)

    if output == 0:
        return render_template('index.html', prediction='THE PATIENT IS NOT LIKELY TO HAVE A HEART FAILURE')
    else:
         return render_template('index.html', prediction='THE PATIENT IS LIKELY TO HAVE A HEART FAILURE')

@app.route('/predict_api',methods = ['POST'])
def predict_api():
     data = request.get_json(force=True)
     prediction = model.predict([np.array(list(data.values()))])

     output = prediction[0]
     return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)

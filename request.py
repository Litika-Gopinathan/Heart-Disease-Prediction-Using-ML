import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'age':37, 'gender':1, 'cp':2, 'trestbps':130, 'chol':250, 'fbs':0,	'restecg':1,	'thalach':187,	'exang':0, 'oldpeak':3.5, 'slope':0, 'ca':0, 'thal':2})

print(r.json())
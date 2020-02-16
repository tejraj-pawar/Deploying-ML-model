import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Avg. Area':800, 'Avg. House Age':9, 'No. of BedRooms':6})

print(r.json())
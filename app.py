from flask import Flask, render_template, request
import pickle
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load model and encoders
model = pickle.load(open('model.pkl', 'rb'))
le_ip = pickle.load(open('ip_encoder.pkl', 'rb'))
le_device = pickle.load(open('device_encoder.pkl', 'rb'))
le_location = pickle.load(open('location_encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    timestamp = request.form['timestamp']
    ip = request.form['ip']
    device = request.form['device']
    location = request.form['location']

    hour = datetime.fromisoformat(timestamp).hour

    try:
        ip_enc = le_ip.transform([ip])[0]
        device_enc = le_device.transform([device])[0]
        location_enc = le_location.transform([location])[0]
    except:
        return render_template('index.html', result="❌ Unknown value detected (IP/Device/Location not seen in training data)")

    data = pd.DataFrame([[hour, ip_enc, device_enc, location_enc]], columns=['hour', 'ip_enc', 'device_enc', 'location_enc'])
    prediction = model.predict(data)

    result = "✅ Normal Login" if prediction[0] == 1 else "🚨 Suspicious Login"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

# Load model and encoders
model = pickle.load(open('model.pkl', 'rb'))
le_ip = pickle.load(open('ip_encoder.pkl', 'rb'))
le_device = pickle.load(open('device_encoder.pkl', 'rb'))
le_location = pickle.load(open('location_encoder.pkl', 'rb'))

@app.route('/predict-login', methods=['POST'])
def predict_login():
    data = request.get_json()
    try:
        # Extract fields
        hour = int(data['hour'])
        ip = le_ip.transform([data['IP']])[0]
        device = le_device.transform([data['device']])[0]
        location = le_location.transform([data['location']])[0]

        # Create input array
        input_data = np.array([[hour, ip, device, location]])
        prediction = model.predict(input_data)[0]

        result = 'suspicious' if prediction == -1 else 'normal'
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

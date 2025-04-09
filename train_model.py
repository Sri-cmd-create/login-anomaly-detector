import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import pickle

# Load login data
df = pd.read_csv('login_data.csv')

# Extract hour from timestamp
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour

# Encode categorical features
le_ip = LabelEncoder()
le_device = LabelEncoder()
le_location = LabelEncoder()

df['ip_enc'] = le_ip.fit_transform(df['IP'])
df['device_enc'] = le_device.fit_transform(df['device'])
df['location_enc'] = le_location.fit_transform(df['location'])

# Features for model
X = df[['hour', 'ip_enc', 'device_enc', 'location_enc']]

# Train Isolation Forest model
model = IsolationForest(contamination=0.15, random_state=42)
model.fit(X)

# Save model and encoders
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('ip_encoder.pkl', 'wb') as f:
    pickle.dump(le_ip, f)
with open('device_encoder.pkl', 'wb') as f:
    pickle.dump(le_device, f)
with open('location_encoder.pkl', 'wb') as f:
    pickle.dump(le_location, f)

print("✅ Model and encoders saved successfully.")

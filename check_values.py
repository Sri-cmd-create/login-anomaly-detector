import pandas as pd

df = pd.read_csv('login_data.csv')
print(df['IP'].unique()[:5])  # Show first 5 IPs

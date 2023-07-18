import pandas as pd
import requests
import streamlit as st

# Load data from x.csv
x_data = pd.read_csv('Sampling.csv', header=None)
x_data.columns = ['Day of the week', 'Temp', 'Hum']

# Download data from Google Drive CSV URL
google_drive_csv_url = 'https://docs.google.com/spreadsheets/d/167nwXzbxfbje4DIWWrl56XV4yktVQekbLnV715GiVEU/export?format=csv'
response = requests.get(google_drive_csv_url)
csv_path = 'google_drive_data.csv'

# Save the downloaded CSV to a local file
with open(csv_path, 'wb') as f:
    f.write(response.content)

# Read the downloaded CSV
google_drive_data = pd.read_csv(csv_path, header=None)

# Remove the column names from the second CSV
#google_drive_data = google_drive_data.iloc[1:]

# Rename the columns of the second CSV
google_drive_data.columns = ['Day of the week', 'Temp', 'Hum']

# Merge the data from x.csv and Google Drive CSV
merged_data = pd.concat([x_data, google_drive_data], ignore_index=True)

# Center align the merged data using CSS styling
centered_data = merged_data.style.set_properties(**{'text-align': 'center'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])

# Display the merged data in Streamlit
st.title("Merged Data")
st.dataframe(merged_data)

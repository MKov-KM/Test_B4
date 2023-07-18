import subprocess
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import time


def load_data():
    data = pd.read_csv("data_electricity.csv")
    data = data[["Day of week", "Hour", "Watt consumed"]]
    data = data.rename({"Day of week": "WeekDay", "Watt consumed": "Watt"}, axis=1)
    data = data.dropna()
    return data

def preprocess_data(data):
    data['Hour'] = data['Hour'].apply(hourasint)
    data = data[data["Watt"].notnull()]
    return data

def hourasint(x):
    if x == '9:00':
        return 9
    if x == '10:00':
        return 10
    if x == '11:00':
        return 11
    if x == '12:00':
        return 12
    if x == '13:00':
        return 13
    if x == '14:00':
        return 14
    if x == '15:00':
        return 15
    if x == '16:00':
        return 16
    if x == '17:00':
        return 17
    if x == '18:00':
        return 18
    if x == '19:00':
        return 19
    if x == '20:00':
        return 20
    return float(x)

def train_model(data):
    X = data.drop("Watt", axis=1)
    X.columns = ["WeekDay", "Hour"]
    y = data["Watt"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    knn_reg = KNeighborsRegressor(n_neighbors=5)
    cv_scores = cross_val_score(knn_reg, X_scaled, y, cv=5, scoring='neg_root_mean_squared_error')
    cv_scores = np.abs(cv_scores)
    knn_reg.fit(X_scaled, y)
    return knn_reg, scaler

def predict_consumption(model, scaler, dow, hour):
    X = np.array([[dow, hour]])
    X = pd.DataFrame(X, columns=["WeekDay", "Hour"])
    X_scaled = scaler.transform(X)
    energy = model.predict(X_scaled)
    return energy[0]

def main():
    st.title("Electricity Consumption Prediction")

    # Load and preprocess the data
    data = load_data()
    data = preprocess_data(data)

    # Train the model
    model, scaler = train_model(data)

    # Input widgets for Day of Week and Hour
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_mapping = dict(zip(days_of_week, range(1, 8)))
    dow = st.selectbox("Day of Week", days_of_week)
    hour = st.slider("Hour", 9, 20, 10)

    if st.button("Calculate Consumption"):
        energy = predict_consumption(model, scaler, dow_mapping[dow], hour)
        st.write("The estimated energy consumption is {:.5f}W".format(energy))

    if st.button("Go to Second Page"):
        subprocess.Popen(["streamlit", "run", "second.py"])
        st.stop()

if __name__ == '__main__':
    main()

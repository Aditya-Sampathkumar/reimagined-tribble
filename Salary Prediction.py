import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("Salary_Data.csv")
print(data.head())
print(data.isnull().sum())

fig = plt.scatter(
    data=data,
    x='Salary',
    y='YearsExperience',
)
plt.show()

x = np.asanyarray(data[["YearsExperience"]])
y = np.asanyarray(data[["Salary"]])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

exp = float(input("Experience in years: "))
features = np.array([[exp]])
print("$", model.predict(features)[0][0])

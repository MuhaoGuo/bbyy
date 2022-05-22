import numpy as np
import pandas as pd

train_raw = pd.read_csv("./data/Train.csv")
test_raw = pd.read_csv("./data/Test.csv")

columns_names =['Date', 'Atmospheric Pressure', 'Minimum Temperature', 'Maximum Temperature', 'Relative Humidity', 'Wind Speed']

print(train_raw.columns)
print(test_raw.columns)
'''
Code to process the age column in the original diabetes dataset
'''
import pandas as pd

path = "../data/diabetes.csv"
df = pd.read_csv(path, sep=',')
col = "age"
age = []

for item in df[col]:
    interval = item[1:-1]
    # print("interval = {}".format(interval))
    split = interval.split("-")
    # print("split is = {}".format(split))
    a = int(split[0])
    b = int(split[1])
    age.append((a + b) / 2)

df[col] = age
df.to_csv("data/diabetes_clean.csv", index=False)

# -*- coding: utf-8 -*-
'''
This is a sample of student code provided for review by General Assembly
https://gist.github.com/jeff-boykin/3af5e25eabd6c10d8aa248c556f625a0

Dataset:
https://gist.github.com/jeff-boykin/9e1a450ef152604e6830ce70f4fc1be8

I edited this code while trying to fix it

'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Load data

url = r'https://gist.githubusercontent.com/jeff-boykin/9e1a450ef152604e6830ce70f4fc1be8/raw/4d42aebc2c2d3f7528a7769248720918e14f2e03/part-2-data.train.csv'
df = pd.read_csv(url)

# Setup data for prediction
x1 = df.SalaryNormalized
x2 = pd.get_dummies(df.ContractType)

# Setup model
model = LinearRegression()

# Evaluate model
#from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split

scores = cross_val_score(model, x2, x1, cv=1, scoring='mean_absolute_error')

print(scores.mean())

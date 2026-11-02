import pandas as pd
import numpy as np
import csv
import cv2 as cv


path = './data/pitches.csv'

df = pd.read_csv(path)

df = df[df.pitch_type != 'UN']
df.dropna(inplace=True)

print(df['pitch_type'])



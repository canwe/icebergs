import json
import sys
from PIL import Image

import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt

# open a CSV file for writing
outF = open('output.csv', 'w')

outF.write('id,is_iceberg\n')

# list of dictionnaries with keys ('id', 'band_1', 'band_2', 'inc_angle', 'is_iceberg')
data_parsed = json.loads(open('train.json').read())
#data_parsed = json.loads(open('test.json').read())

#image = data_parsed[0]['band_1']

def makeApicture(data,file_name):
    min_data = np.min(data)
    max_data = np.max(data)
    for i in range(len(data)):
        data[i] = round((data[i]-min_data)/(max_data-min_data)*255)
    data = np.reshape(data, (75, 75)).T
    data = data.astype('uint8')
    im = Image.fromarray(data)
    im.save(file_name)
#makeApicture(image,"test.jpg")


for image in data_parsed:
    
    makeApicture(image['band_1'],"/Users/lucko625/Desktop/icebergs/train_pics/"+image['id']+"_band_1_"+str(image['is_iceberg'])+".jpg")
    makeApicture(image['band_2'],"/Users/lucko625/Desktop/icebergs/train_pics/"+image['id']+"_band_2_"+str(image['is_iceberg'])+".jpg")
    outF.write(image['id']+','+str(image['is_iceberg'])+'\n')
    #outF.write(image['id']+','+str('0.5')+'\n')

outF.close()

"""
## with pandas
df = pd.read_json('train.json', orient='records')
print(df.shape)
print(df.describe())

df['avg_1'] = np.mean(df['band_1'])
df['avg_2'] = np.mean(df['band_2'])
print(df.describe())
"""



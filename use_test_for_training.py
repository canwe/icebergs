import json
import sys
import csv
import os
from PIL import Image

import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt

DATA_FILENAME='scored_test.json'

score={}
with open('output_unknown_cnn.csv','r') as f:
    input_file=csv.DictReader(f)
    for row in input_file:
        id = row["id"]
        is_iceberg = float(row["is_iceberg"])
        score[id]=round(is_iceberg*100.0)/100.0

""" load the testing data """
# list of dictionnaries with keys ('id', 'band_1', 'band_2', 'inc_angle',) missing 'is_iceberg'
print("starting to load test.json")
data_parsed = json.loads(open('test.json').read())
print("finished loading test.json")
#prescored = pd.read_csv('output_unknown_cnn.csv')

f = open(DATA_FILENAME, 'w')
f.close()

def append_to_json(_dict):
    with open(DATA_FILENAME, 'a+') as f:
        f.seek(0,2)              #Go to the end of file
        if f.tell() == 0 :       #Check if file is empty
            json.dump([_dict], f)  #If empty, write an array
        else:
            f.seek(-1,2)
            f.truncate()           #Remove the last character, open the array
            f.write(' , ')         #Write the separator
            json.dump(_dict,f)     #Dump the dictionary
            f.write(']')           #Close the array


for image in data_parsed:
    is_iceberg = score[image['id']]
    
    entry = {   'id': image['id'],
                'band_1': image['band_1'],
                'band_2': image['band_2'],
                'inc_angle': image['inc_angle'],
                'is_iceberg': is_iceberg,
                }
    append_to_json(entry)



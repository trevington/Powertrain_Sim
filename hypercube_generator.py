import csv
from smt.sampling_methods import LHS
import numpy as np

FILENAME = "points"

# Hypercube bounds
limits = np.array([[0,1],
                  [0,1],
                  [0,1],
                  [0,1]])

sampling = LHS(xlimits=limits)

# Number of points in hypercube
n = 400
x = sampling(n)

'''
# Code to round a variable
for i in range(0, n-1):
    x[i, 0] = np.round(x[i, 0])
'''

x = x.tolist()

# Adds sample number as first csv column
for i in range(0, len(x)):
    x[i] = [i] + x[i]

# Creates sampling csv 
for i in range(0, len(x)):
    with open('{}.csv'.format(FILENAME), 'a', newline = '') as csv_file:
        csvwriter = csv.writer(csv_file)
        csvwriter.writerow(x[i])
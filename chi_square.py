import csv
import numpy as np
from scipy import stats
import pandas as pd
from pandas import crosstab

def data_load():
    x = []
    y = []
    file = './Bestdataset.csv'

    with open(file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        for row in csv_reader:
            x.append(int(row[0]))
            y.append(int(row[34]))
    return np.array(x), np.array(y)

def main():

   X, Y = data_load()
   crosstab = pd.crosstab(X, Y)
   print(crosstab)
   print(stats.chi2_contingency(crosstab)[0:3])
if __name__ == "__main__":
    main()

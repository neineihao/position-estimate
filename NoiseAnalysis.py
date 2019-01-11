import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from errorEstimation import csv2data


def numpy_sort(np_array, arg):
    return np_array[np_array[:, arg].argsort()]

def test():
    data = csv2data("result/result.csv")
    data = numpy_sort(data, 0)
    xData = data[:, 0]
    yData = data[:, 1]

    for i in range(xData.size):
        print("Actual : {}, WithNoise: {}".format(xData[i], yData[i]))
    plt.plot(xData, yData, "*")
    plt.show()

def positionDataProcess():
    data = csv2data("result/RSSdata.csv")
    data = numpy_sort(data, 0)
    row, col = data.shape
    for i in range(row):
        temp = data[i, 0]




if __name__ == '__main__':
    # test()
    positionDataProcess()
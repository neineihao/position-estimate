import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from errorEstimation import csv2data, add_noise, RSS_cal, estimate_distacne, random_test
from os import listdir
from os.path import isfile, join

STD = 0.1
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
    outdata = {}
    data = csv2data("RSSdata.csv")
    data = numpy_sort(data, 0)
    row, col = data.shape
    for i in range(row):
        index = str(round(data[i, 0], 1))
        if index not in outdata:
            outdata[index] = []
            outdata[index].append(data[i, 1::].tolist())
        else:
            outdata[index].append(data[i, 1::].tolist())
    for key, value in outdata.items():
        key = key.replace(".", "_")
        with open("positionData/D{}.csv".format(key), "w") as f:
            f.write("Bxon,Byon,Bzon,Bxoff,Byoff,Bzoff\n")
            for item in value:
                f.write("{},{},{},{},{},{}\n".format(item[0], item[1], item[2], item[3], item[4], item[5]))

def processDistance(filename):
    distance = filename.split(".")[0]
    distance = distance[1::]
    distance = distance.replace("_", ".")
    return float(distance)

def test_distribution():
    path_files = path_file_list()
    for path_file in path_files:
        # test_file = "D49_9.csv"
        data = csv2data("positionData/{}".format(path_file))
        distance = processDistance(path_file)
        print("IN distance: {}".format(distance))
        for item in data:
            on_signal = item[0:3]
            off_signal = item[3:6]
            origin = np.array([[RSS_cal(on_signal, off_signal)]])
            noise_on_signal = add_noise(on_signal)
            noise_off_signal = add_noise(off_signal)
            noise_RSS = np.array([[RSS_cal(noise_on_signal, noise_off_signal)]])
            origin_result = estimate_distacne(origin)
            noise_result = estimate_distacne(noise_RSS)
            print("Origin : {}, Result: {}".format(origin_result,noise_result) )

def path_file_list(path="positionData"):    
    return [f for f in listdir(path) if isfile(join(path, f))]





        
if __name__ == '__main__':
    # test()
    random_test(10000)
    positionDataProcess()
    test_distribution()
    # test_list()

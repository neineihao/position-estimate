import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from errorEstimation import csv2data, add_noise, RSS_cal, estimate_distacne, random_test, draw_histogram
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

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
    data = csv2data("result/RSSdata.csv")
    data = numpy_sort(data, 0)
    row, col = data.shape
    for i in tqdm(range(row),ncols=100, desc="Progress"):
        # for i in range(row):
        # print(data[i,:])
        on_signal = data[i, 1:4]
        # print(on_signal)
        off_signal = data[i, 4:7]
        # print(off_signal)
        origin = np.array([[RSS_cal(on_signal, off_signal)]])
        origin_result = estimate_distacne(origin)
        # print("origin data: {}, data: {}".format(origin_result, data[i,0]))
        # print("abs: {}".format(abs(origin_result - data[i,0])))
        if abs(origin_result - data[i, 0]) < 2:
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

def draw_distribution(test_time):
    path_files = path_file_list()
    with open("DistributionAnalysis.csv", "w") as fi:
        fi.write("Distance, Means, std\n")
        for file_index in tqdm(range(len(path_files)), ncols=100, desc="Progress"):
            # for file_index in tqdm(range(20), ncols=100, desc="Progress"):        
            # file_name = "D145_1"
            data = csv2data("positionData/{}".format(path_files[file_index]))
            ideal = processDistance(path_files[file_index])
            row, col = data.shape
            times = int(test_time / row)
            result_bucket = []
            # for i in tqdm(range(times),ncols=100, desc="Progress"):
            for i in range(times):
                for item in data:
                    on_signal = item[0:3]
                    off_signal = item[3:6]
                    noise_on_signal = add_noise(on_signal)
                    noise_off_signal = add_noise(off_signal)
                    noise_RSS = np.array([[RSS_cal(noise_on_signal, noise_off_signal)]])
                    noise_result = estimate_distacne(noise_RSS)
                    # print("The result: {}".format(noise_result))
                    result_bucket.append(noise_result)
            result_bucket = np.asarray(result_bucket)
            fi.write("{},{},{}\n".format(ideal, result_bucket.mean(), result_bucket.std()))
                    # draw_histogram(result_bucket, xlabel="Estimated Distance")

        # noise_on_signal = add_noise(on_signal)
        # noise_off_signal = add_noise(off_signal)
        # noise_RSS = np.array([[RSS_cal(noise_on_signal, noise_off_signal)]])

        # noise_result = estimate_distacne(noise_RSS)
        # print("Origin : {}, Result: {}".format(origin_result, noise_result))


def test_error(test_time):
    # file_name = "D145_1"
    test_file = "D50_5"
    data = csv2data("positionData/{}.csv".format(test_file))
    ideal = processDistance(test_file)
    row, col = data.shape
    print(data.shape)
    times = int(test_time / row)
    result_bucket = []
    # for i in tqdm(range(times),ncols=100, desc="Progress"):
    for i in range(times):
        for item in data:
            on_signal = item[0:3]
            print(on_signal)
            off_signal = item[3:6]
            noise_on_signal = add_noise(on_signal)
            noise_off_signal = add_noise(off_signal)
            noise_RSS = np.array([[RSS_cal(noise_on_signal, noise_off_signal)]])
            noise_result = estimate_distacne(noise_RSS)
            # print("The result: {}".format(noise_result))
            result_bucket.append(noise_result)
            result_bucket = np.asarray(result_bucket)
    draw_histogram(result_bucket, xlabel="Estimated Distance")

        
if __name__ == '__main__':
    # test()
    # random_test(10000)
    # positionDataProcess()
    # test_distribution()
    # test_list()
    draw_distribution(3000)


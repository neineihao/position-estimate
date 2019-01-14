import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from main import gradient_run, cal_distance_simple
from tqdm import tqdm
import os.path

STD = 0.1


def csv2data(filename='distribution0.csv'):
    df = pd.read_csv(filename, sep=',', index_col=False)
    return np.asarray(df.values)

def test():
    data = csv2data()
    xData = data[:,0]
    yData = data[:,1]
    zData = data[:,2]
    RSSData = data[:,3]
    # draw_histogram(xData, xlabel="Signal")
    # draw_histogram(yData, xlabel="Signal")
    # draw_histogram(zData, xlabel="Signal")
    Xstd = xData.std()
    Ystd = yData.std()
    Zstd = zData.std()
    print("STD in X, Y, Z : ({}, {}, {} )".format(Xstd, Ystd, Zstd))
    draw_histogram(zData, xlabel="Signal")
    print(np.mean(np.array([Xstd, Ystd, Zstd])))

def getMagFlux():
    return random.uniform(-25, 25)

def get3DMag():
    return np.asarray([getMagFlux(), getMagFlux(), getMagFlux()])

def add_noise(flux):
    """
    :param magnetic_flux: numpy shape (3,)
    :return: The numpy array (3,) signal with noise
    """
    noise_flux = np.zeros(flux.shape[0])
    for index, item in enumerate(flux):
        noise_flux[index] = random.gauss(item, STD)
    return noise_flux

def getRSS():
    return RSS_cal(get3DMag(), get3DMag())

def RSS_cal(a, b):
    return (np.sum((a - b) ** 2)) ** 0.5

def get_noise_RSS():
    data = {}
    fluxON = get3DMag()
    fluxOFF = get3DMag()
    data['origin'] = np.array([[RSS_cal(fluxON, fluxOFF)]])
    # fluxNoiseON = add_noise(fluxON)
    # fluxNoiseOFF = add_noise(fluxOFF)
    # data['withNoise'] = np.array([[RSS_cal(fluxNoiseON, fluxNoiseOFF)]])
    data['OnSignal'] = fluxON
    data['OffSignal'] = fluxOFF
    return data

def estimate_distacne(signal, alpha=0.01, times=500, plot=False):
    defaultPosition = np.array([[0,0,0]])
    # print("position: {}, signal: {}".format(defaultPosition, signal))
    position = gradient_run(defaultPosition, signal, alpha=alpha, times=times, plot=plot)
    return position[0]


def generate_data(total_time):
    result = []
    if not os.path.isfile("result/RSSdata.csv"):
        with open("result/RSSdata.csv", "w") as dfile:
            dfile.write("Distance,OnBx,OnBy,OnBz,OffBx,offBy,offBz\n")
    with open("result/RSSdata.csv", "a") as dfile:
        # dfile.write("Distance,OnBx,OnBy,OnBz,OffBx,offBy,offBz\n")
        for i in tqdm(range(total_time), ncols=100, desc="Progress"):
            data = get_noise_RSS()
            origin = data['origin']
            distance = round(cal_distance_simple(origin), 1)
            if distance < 250 and distance > 45:
                result.append(distance)
                dfile.write("{},{},{},{},{},{},{}\n".format(distance,  data['OnSignal'][0], data['OnSignal'][1], data['OnSignal']\
                    	                                          [2], data['OffSignal'][0], data['OffSignal'][1], data['OffSignal'][2]))
    result = np.asarray(result)
    print("From range {} to {}".format(result.max(), result.min()))
    print("Mean Value: {}".format(result.mean()))
    # draw_histogram(result, xlabel="distance")

                


def random_test(total_time):
    result = []
    # with open("result/RSSdata.csv", "a") as dfile:
    with open("RSSdata.csv", "w") as dfile:
        # with open("result/result.csv", "a") as fi:
        dfile.write("Distance,OnBx,OnBy,OnBz,OffBx,offBy,offBz\n")
        for i in tqdm(range(total_time), ncols=100, desc="Progress"):
            data = get_noise_RSS()
            origin = data['origin']
            # withNoise = data['withNoise']
            distance = round(estimate_distacne(origin), 1)
            # print("Signal = {}, Result Distance = {}".format(origin[0, 0], distance))
            # distance_with_noise = round(estimate_distacne(withNoise), 4)
            # print("Signal with Noise = {}, Result Distance with Noise ={}".format(withNoise[0, 0], distance_with_noise))
            if distance < 200 and distance > 50:
                result.append(distance)
                dfile.write("{},{},{},{},{},{},{}\n".format(distance,  data['OnSignal'][0], data['OnSignal'][1], data['OnSignal']\
                    	                                    [2], data['OffSignal'][0], data['OffSignal'][1], data['OffSignal'][2]))
    result = np.asarray(result)
    print("From range {} to {}".format(result.max(), result.min()))
    print("Mean Value: {}".format(result.mean()))
    draw_histogram(result, xlabel="distance")


def test2():
    Bon = np.array([91.67453697, -45.36144034, 83.26887711])
    Bof = np.array([-96.60612616, -56.62736494, -23.10187496])
    RSS = np.array([[RSS_cal(Bon, Bof)]])
    print(RSS)
    distance_data = estimate_distacne(RSS)
    print(distance_data)
    
    
    
def draw_histogram(data, xlabel="RSS", ylabel="Times"):
    plt.hist(data)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    # plt.xlabel('Distance (mm)')
    # plt.ylabel('RSS of Magnetic Flux Density')
    plt.show()


if __name__ == '__main__':
    test2()
    # random_test(100000)
    # add_noise(get3DMag())
    generate_data(10000)


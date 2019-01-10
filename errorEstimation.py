import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from main import gradient_run

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
    return random.uniform(-80, 80)

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
    fluxNoiseON = add_noise(fluxON)
    fluxNoiseOFF = add_noise(fluxOFF)
    data['withNoise'] = np.array([[RSS_cal(fluxNoiseON, fluxNoiseOFF)]])
    return data

def estimate_distacne(signal):
    defaultPosition = np.array([[0,0,0]])
    # print("position: {}, signal: {}".format(defaultPosition, signal))
    position = gradient_run(defaultPosition, signal)
    return position[0]

def random_test():
    result = []
    with open("result.csv", "w") as file:
        file.write(",Origin,Noise\n")
        for i in range(100):
            data = get_noise_RSS()
            origin = data['origin']
            withNoise = data['withNoise']
            distance = round(estimate_distacne(origin), 2)
            # print("Signal = {}, Result Distance = {}".format(origin[0, 0], distance))
            distance_with_noise = estimate_distacne(withNoise)
            # print("Signal with Noise = {}, Result Distance with Noise ={}".format(withNoise[0, 0], distance_with_noise))
            if distance < 200:
                file.write("{}, {}\n".format(distance, distance_with_noise))





    # print("From range {} to {}".format(result.max(), result.min()))
    # print("Mean Value: {}".format(result.mean()))
    # draw_histogram(result, xlabel="distance")


def draw_histogram(data, xlabel="RSS", ylabel="Times"):
    plt.hist(data)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    # plt.xlabel('Distance (mm)')
    # plt.ylabel('RSS of Magnetic Flux Density')
    plt.show()


if __name__ == '__main__':
    # test()
    random_test()
    # add_noise(get3DMag())



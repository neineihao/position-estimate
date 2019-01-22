import numpy as np
import matplotlib.pyplot as plt
from main import RSS_cal
from plot_setting import plot

C = 285692.36935118 * 1.2

def MLE_calculation(position ,signal, times=100, alpha=0.00001, plot=False, color='b'):
    cost_buckets = np.ones(times)
    x = np.ones(times)
    row, col = np.shape(position)
    mu = signal2distance(signal)
    sigma = distance2Noise(mu)
    talpha = alpha
    point = np.array([30.0, 30.0, 30.0])
    for i in range(times):
        if (i % 1000 == 0):
            talpha *= 0.9
        # initial the obj for every iteration
        obj = 0
        for j in range(row):
            cal_result = cal(point, position[j, :], mu[j], sigma[j])
            obj += cal_result['obj']
            point -= talpha * cal_result['grad']
            # print("The cost in iteration {} : {}".format(i, obj))
            cost_buckets[i] = obj
            x[i] = i
        # print("The cost in iteration {} : {}".format(i, obj))
    # print("The point (x, y, z): {}".format(point))
    # print("End for the alpha = {}, and the cost is {}".format(alpha, obj))
    # print("cost: {}".format(obj[0]))
    if plot:
        plt.plot(x, cost_buckets, "{}".format(color), label="alpha = {}".format(alpha))
    return point

def signal2distance(signal):
    return np.power((signal / C), (-1 / 3)) * 4

def distance2Noise(distance):
    # return 0.000001 * np.power(distance, 3) - 0.0002 * np.power(distance, 2) + 0.0161 * distance - 0.4344
    # return 0.0002 * np.power(distance, 2) + 0.0161 * distance - 0.4344
    return 0.000000002 * (distance ** 4)

def cal(target, origin, mu, sigma):
    result = {}
    diff = target - origin
    sigma2 = np.power(sigma, 2)
    k = 1 / (np.power(2 * np.pi * sigma2, 0.5))
    A = np.power(np.power(diff, 2).sum(), 0.5)
    B = A - mu
    C = B / (A * sigma2)
    f = k * np.exp(- np.power(B,2) / (2 * sigma2))
    result['obj'] = - np.log(f)
    result['grad'] = C * diff
    return result

def test():
    # signal = np.array([150.565986, 2.6762, 14.6845])
    # signal = np.array([14.6854, 14.6854, 14.6845])
    # signal = np.array([[2.6762]])
    # signal = np.array([[14.6845]])
    signal = np.array([[150.565986]])
    d = signal2distance(signal)
    noise = distance2Noise(d)
    print(noise)
    position = np.array([[0, 0, 0]])
    result = MLE_calculation(position, signal, plot=True, alpha=0.00001, color="b")
    MLE_calculation(position, signal, plot=True, alpha=0.00003, color="r")
    MLE_calculation(position, signal, plot=True, alpha=0.00005, color="c")
    MLE_calculation(position, signal, plot=True, alpha=0.0001, color="m")
    MLE_calculation(position, signal, plot=True, alpha=0.001, color="y")




    plt.show()
    plt.figure(figsize=(150, 150))
    print("Actual Distance: {}, with estimated: {}".format(d[0], result[0]))
    # print(RSS_cal(result, position[0]))
    # print(RSS_cal(result, position[1]))
    # print(RSS_cal(result, position[2]))
    # print(noise)
    # print(result)

@plot(xl="Iteration(#)", yl="Cost")
def learning_rate_test(signal):
    d = signal2distance(signal)
    noise = distance2Noise(d)
    position = np.array([[0, 0, 0]])
    tuple_list = [(0.1, 'b'), (0.05, 'g'), (0.03, 'r'), (0.01, 'c'), (0.005, 'm'), (0.001, 'k')]
    # tuple_list = [(15, 'b')]

    for item in tuple_list:
        position = np.array([[0, 0, 0]])
        result = MLE_calculation(position, signal, alpha=item[0], plot=True, color=item[1], times=10000)
        print("With learning rate: {}, Distance: {}".format(item[0], result[0]))



if __name__ == '__main__':
    # test()
    mid_signal = np.array([[14.6845]])
    min_signal = np.array([[2.6762838290476405]])
    max_signal = np.array([[150.56598649060965]])
    learning_rate_test(min_signal)


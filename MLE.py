import numpy as np
import matplotlib.pyplot as plt
from main import RSS_cal

C = 285692.36935118 * 1.2

def position_calculation(signal, position, times=1000, alpha=0.001, plot=True, color='b'):
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
            # print(position[1,:])
            # print("mu: {}".format(mu[0]))
            # print("sigma: {}".format(sigma[0]))
            cal_result = cal(point, position[j, :], mu[j], sigma[j])
            obj += cal_result['obj']
            point -= talpha * cal_result['grad']
            print("The cost in iteration {} : {}".format(i, obj))
            cost_buckets[i] = obj
            x[i] = i
        # print("The cost in iteration {} : {}".format(i, obj))
    print("The point (x, y, z): {}".format(point))
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
    # print(f)
    result['obj'] = - np.log(f)
    # print(-np.log(f))
    result['grad'] = C * diff
    return result



def test():
    signal = np.array([150.565986, 2.6762, 14.6845])
    signal = np.array([14.6854, 14.6854, 14.6845])
    # signal = np.array([[2.6762]])
    # signal = np.array([[14.6845]])
    # signal = np.array([[150.565986]])
    d = signal2distance(signal)
    noise = distance2Noise(d)
    print(noise)
    position = np.array([[0, 0, 0]])
    position = np.array([[0, 50, 0], [0, 0, -50], [0, 50, 50]])
    result = position_calculation(signal, position)
    print("Actual Distance: {}, with estimated: {}".format(d[0], result[0]))
    print(RSS_cal(result, position[0]))
    print(RSS_cal(result, position[1]))
    print(RSS_cal(result, position[2]))
    # print(noise)
    # print(result)
    plt.show()




if __name__ == '__main__':
    test()


import numpy as np
import matplotlib.pyplot as plt
import random

A = np.array([[1 / 16], [1 / 16], [1 / 25]])
C = 285692.36935118

def gradient_run(position, signal, color='b', label='test', alpha=0.01, times=500, plot=False):
    cost_buckets = np.ones(times)
    x = np.ones(times)
    row, col = np.shape(position)
    # talpha = alpha / row
    talpha = alpha
    # initial for the estimation result
    point = np.array([float(random.randrange(30,50)), 0.0, 0.0])
    point = np.array([50.0, 0.0, 0.0])
    for i in range(times):
        # initial the obj for every iteration
        obj = 0
        for j in range(row):
            cal_result = calculate(point, position[j, :], signal[j])
            obj += cal_result['obj']
            point -= talpha * cal_result['grad']
        # print("The cost in iteration {} : {}".format(i, obj[0]))
        cost_buckets[i] = obj[0]
        x[i] = i
    # print("The point (x, y, z): {}, {}, {}".format(point[0], point[1], point[2]))
    # print("End for the alpha = {} / N, and the cost is {}".format(alpha, obj[0]))
    #print("cost: {}".format(obj[0]))
    if plot:
        plt.plot(x, cost_buckets, "{}".format(color), label="alpha = {} / N".format(alpha))
    return point


def calculate(cal, position, signal):
    r_dic ={}
    dif = cal - position
    a1 = np.dot(np.power(dif, 2), A)
    a2 = C * np.power(a1, -1.5) - signal
    r_dic['obj'] = np.power(a2, 2)
    grad_co = -3 * C * np.power(a1, -5/2) * a2
    r_dic['grad'] = grad_co * np.multiply(dif, np.array([ 1/8,  1/8, 2/25]))
    return r_dic

def test():
    a = np.array([2, 4, 4])
    b = np.array([1/2, 1/4, 1/4])
    print(np.dot(a, b))

def test_for_distance():
    position = np.array([[0,0,0]])
    signal = np.array([[18.803]])
    result = gradient_run(position, signal, 'b', 'test', alpha=0.001)
    print(result[0])


if __name__ == '__main__':
    """
        'b'	blue
        'g'	green
        'r'	red
        'c'	cyan
        'm'	magenta
        'y'	yellow
        'k'	black
        'w'	white
    """
    # position = np.array([[50.0, 10.0, 20.0]])
    # signal = np.array([[35.401]])
    test_for_distance()
    # position = np.array([[0.0, 0.0, 0.0],[72.6, 0.0, 0.0], [35.829, 49.961, 0.0]])
    # signal = np.array([[19.615],[19.79],[21.27]])
    # gradient_run(position, signal, 'b', 'test', alpha=0.001)
    # gradient_run(position, signal, 'm', 'test', alpha=0.5)
    # gradient_run(position, signal, 'r', 'test', alpha=0.1)
    # gradient_run(position, signal, 'k', 'test', alpha=0.05)
    # gradient_run(position, signal, 'g', 'test', alpha=0.01)
    # gradient_run(position, signal, 'y', 'test', alpha=0.005)
    # gradient_run(position, signal, 'c', 'test', alpha=0.001)
    # plt.xlabel('Iteration')
    # plt.ylabel('Cost')
    # plt.legend()
    # plt.show()

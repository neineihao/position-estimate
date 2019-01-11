import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt



def csv2data(filename='result_z_0.csv'):
    df = pd.read_csv(filename, sep=',', index_col=False)
    r_a = np_sort(df.values)
    # print(r_a)
    return r_a

def main():
    harvest = np.zeros([101,111])
    data = csv2data()
    data[:,1] -= 50
    # print(data[:,1])；
    for row in data:
        harvest[int(row[0]), int(row[1])] = \
            len_cal(row[3:6])
        # angle = angle_cal(row)
        # harvest[int(row[0]), int(row[1])] = angle


    print(harvest.max())
    fig, ax = plt.subplots()
    im = ax.imshow(harvest[0:35,0:35])

    fig.tight_layout()
    plt.xlabel('mm', fontsize=10)
    plt.ylabel('mm', fontsize=10)
    plt.show()

def np_sort(a):
    return a[a[:, 2].argsort()]

def test():
    data = csv2data()
    row, col = data.shape
    distance = data[:,2]
    len_result = np.zeros(row)
    angle_result = np.zeros(row)
    for i in range(row):
        # len_result[i] = len_cal(data[i,3:6])
        angle_result[i] = angle_cal(data[i,:])
    # plt.plot(distance, len_result)
    plt.plot(distance, angle_result)
    plt.ylabel("Rotation (radian)")
    plt.xlabel("Distance (mm)")
    # plt.xlabel('Distance (mm)')
    # plt.ylabel('RSS of Magnetic Flux Density')
    plt.show()

def len_cal(data):
    return  (data[0] ** 2 + data[1] ** 2 + data[2] ** 2) ** 0.5

def angle_cal(row):
    ref_len = 13.6189
    ref_vector = np.array([9.600146, -5.159825, -8.166265])
    value = np.dot(row[3:6], ref_vector) / (len_cal(row[3:6]) * ref_len)
    result = np.arccos(value)
    return result


if __name__ == '__main__':
    # print("Hello World")
    # main()
    test()
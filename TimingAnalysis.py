import numpy as np
import matplotlib.pyplot as plt

def pre_data(filename='test.txt'):
    with open("test.txt", "r") as f:
        container = []
        for line in f:
            line.strip()
            container.append(int(line))
    data = np.asarray(container)
    return data

def draw_histogram(data):
    plt.hist(data)
    plt.xlabel("time (ms)")
    plt.ylabel("counts (#)")
    plt.show()

def main():
    data = pre_data()
    print(data.mean())
    draw_histogram(data)


if __name__ == '__main__':
    main()

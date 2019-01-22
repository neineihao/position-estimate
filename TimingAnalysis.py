import numpy as np
import matplotlib.pyplot as plt
from plot_setting import plot
from main import csv2data

def pre_data(filename='test.txt'):
    with open("test.txt", "r") as f:
        container = []
        for line in f:
            line.strip()
            container.append(int(line))
    data = np.asarray(container)
    return data

@plot(xl="Magnetic Flux Density(mT)", yl="Counts(#)")
def draw_histogram(data):
    plt.hist(data)
    # plt.xlabel("time (ms)")
    # plt.ylabel("counts (#)")
    # plt.show()



def main():
    data = csv2data()
    draw_histogram(data[:,3])


if __name__ == '__main__':
    main()

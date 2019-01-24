import numpy as np
from main import RSS_cal
from MLE import MLE_calculation
from main import gradient_run
import matplotlib.pyplot as plt
from tqdm import tqdm

C = 285692.36935118 * 1.2
NoiseSTD = 0.1

def generate_position():
    position_set = []
    for z in range(50 ,100, 5):
        for y in range(-50, 50, 5):
            for x in range(-50, 50, 5):
                position_set.append([x, y, z])
    return np.asarray(position_set)

def get_distance_data(signal_sources):
    result = {}
    position = generate_position()
    row, col = position.shape
    s_row, s_col = signal_sources.shape
    result['distance'] = np.ones([row, s_row])
    result['position'] = position
    for i in range(row):
        for j in range(s_row):
            result['distance'][i][j] = RSS_cal(signal_sources[j,:], position[i, :])
    return result

def signal2distance(signal):
    return np.power(signal / C, -1 / 3) * 4

def distance2signal(distance):
    return np.power(4 / distance, 3) * C

def process(test_times=10):
    # signal_sources = np.array([[0, 0, 0], [50, 0, 0], [50, 50, 0], [0, 50, 0]])
    str_numpy = lambda np_array : ", ".join(map(str, np_array.tolist()))
    signal_sources = np.array([[0, 0, 0], [50, 0, 0], [50, 50, 0]])
    finger_print_data = get_distance_data(signal_sources)
    row, col = finger_print_data['distance'].shape
    distance_matrix = finger_print_data['distance']
    signal_matrix = distance2signal(distance_matrix)
    # result = np.ones([3,row])
    with open("EstResult/TriResult.csv", "w") as Tri_F:
        with open("EstResult/MLEResult.csv", "w") as MLE_F:
            Tri_F.write("OriginX, OriginY, OriginZ, EstX, EstY, EstZ\n")
            MLE_F.write("OriginX, OriginY, OriginZ, EstX, EstY, EstZ\n")
            for index in tqdm(range(row), ncols=100, desc="Progress"):
                origin = finger_print_data['position'][index, :]
                MLE_F.write(str_numpy(origin) + ", ")
                Tri_F.write(str_numpy(origin) + ", ")
                MLE_position, triangulation = np.zeros(3), np.zeros(3)
                for times in range(test_times):
                    test_signal = add_noise(signal_matrix[index, :])
                    MLE_position += MLE_calculation(signal_sources, test_signal, alpha=0.003, times=8000)
                    triangulation += gradient_run(signal_sources, test_signal, alpha=0.05, times=8000)
                average_sample = lambda x: np.round(x/test_times, 2)
                MLE_position = average_sample(MLE_position)
                MLE_F.write(str_numpy(MLE_position) + "\n")
                triangulation = average_sample(triangulation)
                Tri_F.write(str_numpy(triangulation) + "\n")
                # print("The origin point: {}".format(origin))




def add_noise(signal):
    return np.random.normal(signal, NoiseSTD)

def test():
    result = []
    signal_sources = np.array([[0, 0, 0], [50, 0, 0], [50, 50, 0]])
    get_distance_data(signal_sources)

    # signal = np.array([150.56598649060965])
    # distance = signal2distance(signal)
    # print(distance)
    # r_signal = distance2signal(np.array([86.6025, 86.6025, 86.6025, 86.6025]))
    signal = np.array([86.6025, 86.6025, 86.6025, 86.6025])
    signal = np.array([2.6762838290476405, 2.6762838290476405, 2.6762838290476405, 2.6762838290476405])
    distance = signal2distance(signal)
    noise_signal = add_noise(signal)
    # print(noise_signal)
    r_distance = signal2distance(noise_signal)
    result.append([])
    print(distance)
    print(r_distance)


    # process()



if __name__ == '__main__':
    # test()
    process()

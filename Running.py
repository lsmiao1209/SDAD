import time
import os

import Absolution.Nauxiliary
from metrics.sdhots import exhibit_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
seed = 42
from SDADT.main import launch_SDAD
from Tool.utils import *

if __name__ == '__main__':
    seed_torch(seed)
    starttime = time.time()

    train = 0
    for dataname in ['vertebral', 'annthyroid', 'pima', 'WPBC', 'waveform', 'speech', 'CIFAR10_0', 'amazon', 'imdb', '20news_0']:
        seed_torch(seed)
        print(dataname)
        T = 500
        lamda = 300
        train_x, train_y, test_x, test_y = getdata(dataname)
        if train == 0:
            print('AnoD/AnoS/AnoSD')
            Absolution.Nauxiliary.launch_DAD(dataname, train_x, train_y, test_x, test_y, T, lamda)
            launch_SDAD(dataname, train_x, train_y, test_x, test_y, T, lamda)
        else:
            Compared.PyOD.compare(dataname, train_x, train_y, test_x, test_y)

    time_taken = time.time() - starttime
    print("Running Time：", time_taken)
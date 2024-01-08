# https://github.com/yzhao062/pyod
import os
import time
import numpy as np
import pandas as pd
import sklearn
from pyod.models.vae import VAE
from pyod.models.alad import ALAD
from pyod.models.anogan import AnoGAN
from pyod.models.mo_gaal import MO_GAAL
from Compared.DTPM import DTECategorical
from Compared.DDPM import DDPM
from deepod.models.tabular import DeepIsolationForest
from deepod.models.tabular import ICL
from adbench.baseline.GANomaly.run import GANomaly

from sklearn.metrics import roc_auc_score
from sklearn import metrics

from Tool.utils import get_err_threhold

def compare(dataname, train_x,train_y, test_x, test_y):
    output_file = f'./metrics/data/pyodresult.xlsx'
    if os.path.exists(output_file):
        df = pd.read_excel(output_file)
        df = pd.concat([df, pd.DataFrame([dataname])], ignore_index=True)
        df.to_excel(output_file, index=False)
    else:
        print("fdasfffff")

    maxauc = 0.0
    maxpr = 0.0
    maxf1 = 0.0

    maxsample = 256
    if train_x.shape[0] < maxsample:
        maxsample = train_x.shape[0]

    if test_x.shape[1] < 64:
        VAEneurons = [16, 8, 4], [4, 8, 16]
    else:
        VAEneurons = [128, 64, 32], [32, 64, 128]


    classifiers = {
        # tips tensorflow: vae 需要跑参数时，在本地跑
        # tips: 计算模型参数量，需要在本地环境跑，因为修改了pyod的文件
        # 'VAE': VAE(batch_size=100, encoder_neurons=VAEneurons[0],decoder_neurons=VAEneurons[1]),
        'AnoGAN': AnoGAN(),
        # 'ALAD': ALAD(),
        # 'GANomaly':GANomaly(seed=42),
        # 'MO_GAAL': MO_GAAL(k=10),
        # 'DDPM': DDPM(),
        'DTPM': DTECategorical(),
        # 'ICL': ICL(),
        # 'DIF': DeepIsolationForest(max_samples=maxsample)

    }
    name = []
    maxauc_list =  []
    maxpr_list =  []
    maxf1_list =  []

    for clf_name, clf in classifiers.items():
        print(f"Using {clf_name} method")
        starttime = time.time()

        if clf_name in ['DTPM']:
            clf = DTECategorical()
            clf.fit(train_x)
            st = time.time()
            test_scores = clf.predict_score(test_x)
            auc, pr, f1 = Metrics(test_y, test_scores)
            if auc > maxauc:
                maxauc = auc
                maxpr = pr
                maxf1 = f1
            print('{:.4f}'.format(time.time() - st))
        elif clf_name in ['DDPM']:
            clf = DDPM()
            clf.fit(train_x)
            st = time.time()
            test_scores = clf.predict_score(test_x)
            auc, pr, f1 = Metrics(test_y, test_scores)
            if auc > maxauc:
                maxauc = auc
                maxpr = pr
                maxf1 = f1
            print('{:.4f}'.format(time.time() - st))

        elif clf_name in ['GANomaly']:
            clf = GANomaly(seed=42)

            clf.fit(train_x,train_y)

            st = time.time()
            test_scores = clf.predict_score(test_x)
            auc, pr, f1 = Metrics(test_y, test_scores)
            if auc > maxauc:
                maxauc = auc
                maxpr = pr
                maxf1 = f1
            print('{:.4f}'.format(time.time() - st))
        else:
            clf.fit(train_x)
            st = time.time()

            test_scores = clf.decision_function(test_x)
            auc, pr, f1 = Metrics(test_y, test_scores)
            maxauc = auc
            maxpr = pr
            maxf1 = f1

            print('{:.4f}'.format(time.time() - st))
        time_taken = time.time() - starttime

        # 展示数据
        name.append(clf_name)
        maxauc_list.append(maxauc)
        maxpr_list.append(maxpr)
        maxf1_list.append(maxf1)
        maxauc = 0.0
        maxpr = 0.0
        maxf1 = 0.0

    # for idx, metr in zip(['AUC', 'PR', 'F1'], [maxauc_list, maxpr_list, maxf1_list]):
    #     print(idx+':')
    #     for i in metr:
    #         print('%.4f' %i)

    for nam, auc, pr, f1 in zip(name, maxauc_list, maxpr_list, maxf1_list):
        result_str = '{}, {:.4f}, {:.4f}'.format(nam, auc, pr)
        print(result_str)
        output_file = f'./metrics/data/pyodresult.xlsx'
        if os.path.exists(output_file):
            df = pd.read_excel(output_file)
            df = pd.concat([df, pd.DataFrame([result_str])], ignore_index=True)
            df.to_excel(output_file, index=False)
        else:
            print("fdasfffff")






def Metrics(test_y, error):
    auc = roc_auc_score(test_y, error)
    pr = sklearn.metrics.average_precision_score(test_y, error)

    fpr, tpr, thresholds = metrics.roc_curve(test_y, error, pos_label=1)
    dr, far, best_th, _ = get_err_threhold(fpr, tpr, thresholds)
    test_labels = np.where(error > best_th, 1, 0)
    f1 = metrics.f1_score(test_y, test_labels)

    return auc, pr, f1
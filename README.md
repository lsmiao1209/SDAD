#SDAD

## Software Requirement
```
numpy== 1.25.0
torch==1.13.0+cu116
python==3.9.18
sklearn==0.22.0
pandas==2.1.3
cuda=11.6
pyod==1.0.9
adbench==0.1.11
deepod==0.4.1
```
## Datasets
To demonstrate the effectiveness of our proposed method, we conducted experiments on ten real-world datasets. Five of these datasets are of lower dimension ('vertebral', 'annthyroid', 'pima', 'WPBC', 'waveform'), while the remaining five belong to higher dimensions('speech', 'CIFAR10_0', 'amazon', 'imdb', '20news_0'). All datasets are derived from a recent benchmark study of anomaly detection, adbench (https://github.com/Minqi824/ADBench/tree/main/adbench/datasets).

## Baseline methods
To validate the performance of our proposed AnoSD, we conducted a comparative evaluation with nine state-of-the-art methods on ten datasets. The Baseline methods include generative model-based approaches such as VAE[2013], AnoGAN[2017], ALAD[2017], GANomaly[2018], MOGAAL[2019], DDPM[2020], and DTPM[2023]. Additionally, two recent approaches were considered: one based on self-supervised learning, ICL[2022], and another is on deep ensemble framework, DIF[2023].


## Evaluation measures
For the evaluation metrics, we utilized two mainstream methods widely employed in anomaly detection\cite{dtpm,slad,dif,metrics}: the Area Under the Receiver Operating Characteristic Curve (AUC-ROC) and the Area Under the Precision Recall Curve (AUC-PR).

## Train the model
To obtain AnoSD's performance and its variants on a speicifc dataset, run running.py and set train=0;

To obtain the baseline methods' performance on a speicifc dataset, run running.py and set train=1;

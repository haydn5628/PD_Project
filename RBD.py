import os
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
from tqdm import tqdm, trange
from mySSA import mySSA
from sklearn.ensemble import RandomForestClassifier


process_address = 'C:/Users/hayde/OneDrive/桌面/Normal(all).csv'
All = pd.read_table(process_address, engine = 'python',sep = ";")
address = "C:/Users/hayde/OneDrive/桌面/muscle.csv"
musle = pd.read_table(address, engine = 'python',sep = ";")
stage_address = "D:/school/AIGO/data/PD與正常病人的匯出資料/PD與正常病人的匯出資料/Normal/ASCII/normal(Hypnogram).TXT"
stage_list = pd.read_table(stage_address, engine = 'python',sep = ";").T.reset_index(drop=True).T


def feature_SSA(time_series):
    ssa = mySSA(time_series)
    ssa.embed(embedding_dimension=int(len(time_series)/3), suspected_frequency=1, verbose=False)
    ssa.decompose(verbose=False)
    return [ssa.s[i] for i in range(5)]

#######################
print('Train model')
RF = None 
RF = RandomForestClassifier(n_estimators=2500, verbose=0)
results = cross_val_score(RF, np.array(train_feature), np.array(train_label), cv=KFold(10, random_state=True, shuffle=True))
print(results, results.mean(), results.std())
RF.fit(np.array(train_feature), np.array(train_label))


REM_list = [i for i,x in enumerate(stage_list.values) if x=="R"]


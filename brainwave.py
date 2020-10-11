import os
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
from tqdm import tqdm, trange

brainwavedata_address = 'D:/school/AIGO/data/PD與正常病人的匯出資料/PD與正常病人的匯出資料/Normal/ASCII/腦波訊號'
brainwavedata_TXTs = os.listdir(brainwavedata_address)

tStart = time.time()#計時開始
C3 = pd.read_table(brainwavedata_address + '/' + brainwavedata_TXTs[0],engine='python').T.reset_index(drop=True).T # C3
tEnd = time.time()#計時結束
print("It cost %f sec" % (tEnd - tStart))
C4 = pd.read_table(brainwavedata_address + '/' + brainwavedata_TXTs[2],engine='python').T.reset_index(drop=True).T # C4
F3 = pd.read_table(brainwavedata_address + '/' + brainwavedata_TXTs[8],engine='python').T.reset_index(drop=True).T # F3
F4 = pd.read_table(brainwavedata_address + '/' + brainwavedata_TXTs[9],engine='python').T.reset_index(drop=True).T # F4
M1 = pd.read_table(brainwavedata_address + '/' + brainwavedata_TXTs[10],engine='python').T.reset_index(drop=True).T # M1
M2 = pd.read_table(brainwavedata_address + '/' + brainwavedata_TXTs[11],engine='python').T.reset_index(drop=True).T # M2

f4_m1 = F4 - M1
f3_m2 = F3 - M2
c4_m1 = C4 - M1
c3_m1 = C3 - M1

plot = plt.figure()
plt.subplot(4,1,1)
plt.title("E1-M2, before erosion filter")
plt.plot(f4_m1[1000000:6000000], linewidth = 0.5)
plt.subplot(4,1,2)
plt.title("E1-M2, after erosion filter")
plt.plot(f3_m2[1000000:6000000], linewidth = 0.5)
plt.subplot(4,1,3)
plt.title("E1-M2, after erosion filter")
plt.plot(c4_m1[1000000:6000000], linewidth = 0.5)
plt.subplot(4,1,4)
plt.title("E1-M2, after erosion filter")
plt.plot(c3_m2[1000000:6000000], linewidth = 0.5)
plt.show()

sample = f4_m1[1000000:1007680]
y = list()
sample.iloc[1,0]
for i in tqdm(range(len(sample))):
    y.append(min(sample.iloc[i:i+4,0]))

# Lanczos Differentiation Filter
w = list()
for i in range(len(y)-15):
    w.append(0.5 * (y[i+15] - y[i]))

# Median Filter 中值濾波
m = list()
for i in range(len(w)-15):
    m.append(statistics.median(w[i:i+15]))


for l in range(100):
    start = 1000000 + l * 7680
    end = start + 7680
    sample = f4_m1[start:end]
    i=0
    j=0
    k=0
    # Linear Erosion Filter
    y = list()
    for i in range(len(sample)):
        y.append(min(sample.iloc[i:i+4,0]))
    # Lanczos Differentiation Filter
    w = list()
    for j in range(len(y)-15):
        w.append(0.5 * (y[j+15] - y[j]))
    # Median Filter 中值濾波
    m = list()
    for k in range(len(w)-15):
        m.append(statistics.median(w[k:k+15]))
    plot = plt.figure()
    plt.subplot(4,1,1)
    plt.title("f4_m1, before erosion filter")
    plt.plot(sample, linewidth = 0.5)
    plt.subplot(4,1,2)
    plt.title("f4_m1, after erosion filter")
    plt.plot(y, linewidth = 0.5)
    plt.subplot(4,1,3)
    plt.title("f4_m1, after 1st order Lanczos differentiation filter")
    plt.plot(w, linewidth = 0.5)
    plt.subplot(4,1,4)
    plt.title("f4_m1, after Median filter")
    plt.plot(m, linewidth = 0.5)
    plt.show()


for i in range(1000):
    start = 998400 + i * 7680
    end = start + (i+1) * 7680
    sample = C3_M1.iloc[start:end,4]
    plot = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(c3_m1[start:end],linewidth = 0.5)
    plt.title("C3_M1, before correction." + stage.iloc[131 + i,0])
    plt.subplot(2,1,2)
    plt.plot(sample,linewidth = 0.5)
    plt.title("C3_M1, after correction.")
    plt.show()


process_address = 'C:/Users/hayde/OneDrive/桌面/Normal(all).csv'
process_address = 'C:/Users/hayde/OneDrive/桌面/C4_M1.csv'
C3_M1 = pd.read_table(process_address, engine = 'python',sep = ";")
C3_M1.iloc[:,4]



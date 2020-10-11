import os
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics

brainwavedata_address = 'D:/school/AIGO/data/PD與正常病人的匯出資料/PD與正常病人的匯出資料/Normal/ASCII/腦波訊號'
brainwavedata_TXTs = os.listdir(brainwavedata_address)

CH1 = pd.read_table(brainwavedata_address + '/' + brainwavedata_TXTs[3],engine='python').T.reset_index(drop=True).T # Chin1
CH2 = pd.read_table(brainwavedata_address + '/' + brainwavedata_TXTs[4],engine='python').T.reset_index(drop=True).T # Chin2
CH3 = pd.read_table(brainwavedata_address + '/' + brainwavedata_TXTs[5],engine='python').T.reset_index(drop=True).T # Chin3

ch1_ch2 = CH1 - CH2
ch1_ch3 = CH1 - CH3
ch2_ch3 = CH2 - CH3

start = 506880
end = 514560
sample = ch1_ch2[start:end]
sample1 = ch1_ch2[start:end]

plot = plt.figure()
plt.subplot(2,1,1)
plt.title("E1-M2, before erosion filter")
plt.plot(CH1[:], linewidth = 0.5)
plt.subplot(2,1,2)
plt.title("E1-M2, after erosion filter")
plt.plot(ch1_ch3[1000000:6000000], linewidth = 0.5)
plt.show()


for l in range(100):
    start = 1000000 + l * 7680
    end = start + 7680
    sample = ch1_ch2[start:end]
    # eye blink detection algorithm
    BlinkDuration = 200 # miliseconds 100-400
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
    plt.title("E1-M2, before erosion filter")
    plt.plot(sample, linewidth = 0.5)
    plt.subplot(4,1,2)
    plt.title("E1-M2, after erosion filter")
    plt.plot(y, linewidth = 0.5)
    plt.subplot(4,1,3)
    plt.title("E1-M2, after 1st order Lanczos differentiation filter")
    plt.plot(w, linewidth = 0.5)
    plt.subplot(4,1,4)
    plt.title("E1-M2, after Median filter")
    plt.plot(m, linewidth = 0.5)
    plt.show()

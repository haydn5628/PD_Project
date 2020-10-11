import os
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import statistics

brainwavedata_address = 'D:/school/AIGO/data/PD與正常病人的匯出資料/PD與正常病人的匯出資料/Normal/ASCII/腦波訊號'
brainwavedata_TXTs = os.listdir(brainwavedata_address)


E1 = pd.read_table(brainwavedata_address + '/' + brainwavedata_TXTs[6],engine='python').T.reset_index(drop=True).T # E1
E2 = pd.read_table(brainwavedata_address + '/' + brainwavedata_TXTs[7],engine='python').T.reset_index(drop=True).T # E2
M2 = pd.read_table(brainwavedata_address + '/' + brainwavedata_TXTs[11],engine='python').T.reset_index(drop=True).T # M2

e1_m2 = E1 - M2
e2_m2 = E2 - M2

# for i in range(50,100):
# start = i * 7680
# end = 7680 * (i+1)
# plot = plt.figure()
# plt.subplot(2,1,1)
# plt.title("E1-M2, " + "page "+ str(i*1) + " to " + str(i*1+1))
# plt.plot(e1_m2[start:end], linewidth = 0.1)
# plt.subplot(2,1,2)
# plt.title("E2-M2")
# plt.plot(e2_m2[start:end], linewidth = 0.1)
# plt.show()

start = 506880
end = 514560
sample = e1_m2[start:end]

# eye blink detection algorithm
BlinkDuration = 200 # miliseconds 100-400

# Linear Erosion Filter
y = list()
sample.iloc[1,0]
for i in range(len(sample)):
    y.append(min(sample.iloc[i:i+4,0]))

# Lanczos Differentiation Filter
w = list()
for i in range(len(y)-15):
    w.append(0.5 * (y[i+15] - y[i]))

# Median Filter 中值濾波
m = list()
for i in range(len(w)-15):
    m.append(statistics.median(w[i:i+15]))

# Threshold Filter 設定閾值
z = list()
for i in range(len(m)-30):
    if abs(m[i+15]) >= 0.000015:#abs(statistics.median(w[i:i+30])):
        z.append(m[i+15])
    else:
        z.append(0)

plot = plt.figure()
plt.subplot(5,1,1)
plt.title("E1-M2, before erosion filter")
plt.plot(sample[0+2:7680-2], linewidth = 0.5)
plt.subplot(5,1,2)
plt.title("E1-M2, after erosion filter")
plt.plot(y[0:7680], linewidth = 0.5)
plt.subplot(5,1,3)
plt.title("E1-M2, after 1st order Lanczos differentiation filter")
plt.plot(w[0:7680-2], linewidth = 0.5)
plt.subplot(5,1,4)
plt.title("E1-M2, after Median filter")
plt.plot(m[0:7680-10], linewidth = 0.5)
plt.subplot(5,1,5)
plt.title("E1-M2, after threshold filter")
plt.plot(z[0:7680-10], linewidth = 0.5)
plt.show()




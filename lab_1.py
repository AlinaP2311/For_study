# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 19:59:48 2021

@author: Alina
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


# 1. Загрузить датасет в датафрейм, и исключить бинарные признаки и
#признак времени.

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
print(df)
df = df.drop(columns=['anaemia','diabetes','high_blood_pressure','sex','smoking','time','DEATH_EVENT'])
print(df)

#2. 
n_bins = 20
fig, axs = plt.subplots(2, 3)
axs[0, 0].hist(df['age'].values,bins=n_bins)
axs[0, 0].set_title('Возраст')

axs[0, 1].hist(df['creatinine_phosphokinase'].values, bins=n_bins)
axs[0, 1].set_title('Креатинкиназа')

axs[0, 2].hist(df['ejection_fraction'].values, bins=n_bins)
axs[0, 2].set_title('фракция выброса')

axs[1, 0].hist(df['platelets'].values, bins=n_bins)
axs[1, 0].set_title('тромбоциты')

axs[1, 1].hist(df['serum_creatinine'].values, bins=n_bins)
axs[1, 1].set_title('креатинин сыворотки')

axs[1, 2].hist(df['serum_sodium'].values, bins=n_bins)
axs[1, 2].set_title('сывороточный натрий')

plt.ion()
plt.show()
                   
data=df.to_numpy(dtype='float')

#scaler = sc.preprocessing.StandardScaler().fit(data[:150,:])

scaler = preprocessing.StandardScaler().fit(data[:150,:])
data_scaled = scaler.transform(data)

fig, axs = plt.subplots(2,3)
axs[0, 0].hist(data_scaled[:,0], bins = n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_scaled[:,1], bins = n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_scaled[:,2], bins = n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_scaled[:,3], bins = n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_scaled[:,4], bins = n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_scaled[:,5], bins = n_bins)
axs[1, 2].set_title('serum_sodium')
plt.ion()
plt.show()














import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model


# Load data
chara_data = pd.read_csv("your Get_feature.csv path")  
weight_data = np.genfromtxt("your path//Scale.csv", dtype='float', delimiter=',') 

# Get feature from previous file
ID = chara_data['ID']
area_data = chara_data['area']
peri_data = chara_data['peri']
length_data = chara_data['length']
# This includes weight of 125 shrimps in data folder
# You only focus on the last row of file.
weight = weight_data[3][25:125].T 
# take 100 shrimps from 25 to 125 fro training and the ramain for testing
# area
area = area_data[25:125].T
area = np.array(area).reshape(area.shape[0],1)
one1 = np.ones((area.shape[0], 1))
area_bar = np.concatenate((one1, area), axis = 1)
# perimeter
peri = peri_data[25:125].T
peri = np.array(peri).reshape(peri.shape[0],1)
one2 = np.ones((peri.shape[0], 1))
peri_bar = np.concatenate((one2, peri), axis = 1)
# length
length = length_data[25:125].T
length = np.array(length).reshape(length.shape[0],1)
one3 = np.ones((length.shape[0], 1))
length_bar = np.concatenate((one3, length), axis = 1)

# area_peri
area_peri = np.concatenate((area, peri), axis = 1)
area_peri = np.array(area_peri).reshape(area_peri.shape[0],2)
area_peri_bar = np.concatenate((one1,area_peri), axis = 1)


# training area model
regr = linear_model.LinearRegression(fit_intercept=False) 
regr.fit(area_bar, weight)
# coefficient
w = regr.coef_
print(w)

# test data
# area_test
area_test = area_data[0:25].T
# area_test = area_test/max(area_data)
area_test = np.array(area_test).reshape(area_test.shape[0],1)
one1_test = np.ones((area_test.shape[0], 1))
area_bar_test = np.concatenate((one1_test, area_test), axis = 1)
# perimeter_test
peri_test = peri_data[0:25].T
peri_test = np.array(peri_test).reshape(peri_test.shape[0],1)
one2_test = np.ones((peri_test.shape[0], 1))
peri_bar_test = np.concatenate((one2_test, peri_test), axis = 1)
# length_test
length_test = length_data[0:25].T
length_test = np.array(length_test).reshape(length_test.shape[0],1)
one1_test = np.ones((length_test.shape[0], 1))
length_bar_test = np.concatenate((one1_test, length_test), axis = 1)

# area_peri_test
area_peri_test = np.concatenate((area_test, peri_test), axis = 1)
area_peri = np.array(area_peri).reshape(area_peri.shape[0],2)
area_peri_bar_test = np.concatenate((one1_test,area_peri_test), axis = 1)

#test predict

weight_predict = regr.predict(area_bar_test)

for i in range (len(weight_predict)):
    print(weight_predict[i])
    # f.write("{} \n".format(weight_predict[i]))
# print((regr.predict(area_bar_test)))
weight_test = weight_data[3][0:25].T
# plot data
fig,ax = plt.subplots()
ax.scatter(area,weight,label = 'data')
ax.scatter(area_test,weight_test, c='g',label = 'test')
# Drawing the fitting line
w_0 = w[0]
w_1 = w[1]
print(w_0)
print(w_1)
x0 = np.linspace(80000, 300000, 2000)
y0 = w_0 + w_1*x0
# plot fitting line
ax.plot(x0, y0,'r', label = 'line')
plt.xlabel('area')
plt.ylabel('weight')
legend = ax.legend(shadow= False)
plt.show()

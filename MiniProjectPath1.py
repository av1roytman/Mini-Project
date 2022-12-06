import numpy as np
import pandas
import pandas as pd
from matplotlib import pyplot as plt
import statistics
import statsmodels.api as stats

''' 
The following is the starting code for path1 for data reading to make your first step easier.
'dataset_1' is the clean data for path1.
'''
dataset_1 = pandas.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
dataset_1['Brooklyn Bridge'] = pandas.to_numeric(dataset_1['Brooklyn Bridge'].replace(',', '', regex=True))
dataset_1['Manhattan Bridge'] = pandas.to_numeric(dataset_1['Manhattan Bridge'].replace(',', '', regex=True))
dataset_1['Queensboro Bridge'] = pandas.to_numeric(dataset_1['Queensboro Bridge'].replace(',', '', regex=True))
dataset_1['Williamsburg Bridge'] = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',', '', regex=True))
dataset_1['Williamsburg Bridge'] = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',', '', regex=True))
# print(dataset_1.to_string()) #This line will print out your data

# PROBLEM 1

brooklyn = list(dataset_1['Brooklyn Bridge'])
brooklyn = [int(x) for x in brooklyn]

manhattan = list(dataset_1['Manhattan Bridge'])
manhattan = [int(x) for x in manhattan]

williamsburg = list(dataset_1['Williamsburg Bridge'])
williamsburg = [int(x) for x in williamsburg]

queensboro = list(dataset_1['Queensboro Bridge'])
queensboro = [int(x) for x in queensboro]

x = [0] * 214
for i in range(214):
    x[i] = i + 1

fig, pos = plt.subplots(2, 2)
fig.suptitle("Number of bikes in each city per day")
pos[0, 0].plot(x, brooklyn, color='red')
pos[0, 0].set_title('Brooklyn')
pos[0, 0].set(xlim=(0, 214), ylim=(0, 10000))
pos[0, 0].set(xlabel='Number of Day', ylabel='Number of bikes')

pos[1, 0].plot(x, manhattan, color='blue', label='Manhattan')
pos[1, 0].set(xlim=(0, 214), ylim=(0, 10000))
pos[1, 0].legend(loc='upper right')
pos[1, 0].set(xlabel='Number of Day', ylabel='Number of bikes')

pos[0, 1].plot(x, williamsburg, color='green')
pos[0, 1].set_title('Williamsburg')
pos[0, 1].set(xlim=(0, 214), ylim=(0, 10000))
pos[0, 1].set(xlabel='Number of Day', ylabel='Number of bikes')

pos[1, 1].plot(x, queensboro, color='black', label='Queensboro')
pos[1, 1].set(xlim=(0, 214), ylim=(0, 10000))
pos[1, 1].legend(loc='upper right')
pos[1, 1].set(xlabel='Number of Day', ylabel='Number of bikes')

plt.show()

# PROBLEM 2

high = list(dataset_1['High Temp'])
low = list(dataset_1['Low Temp'])
total = list(dataset_1['Total'])
total = [i.replace(",", "") for i in total]
total = [float(x) for x in total]

plt.scatter(high, total, color="blue")
plt.title('Total Riders vs High Temp')
plt.xlabel('High Temp')
plt.ylabel('Total Riders')
plt.grid(True)
plt.show()

plt.scatter(low, total, color="red")
plt.title('Total Riders vs Low Temp')
plt.xlabel('Low Temp')
plt.ylabel('Total Riders')
plt.grid(True)
plt.show()

X = dataset_1[['High Temp', 'Low Temp']]
Y = total

X1 = stats.add_constant(X)
model = stats.OLS(Y, X1)
results = model.fit()
print("params: \n", results.params)
print("R squared:", results.rsquared)

precipitation = list(dataset_1['Precipitation'])
for i in range(len(precipitation)):
    if precipitation[i] == 'T':
        precipitation[i] = 0
precipitation[3] = 0.47
precipitation = [float(x) for x in precipitation]

plt.scatter(total, precipitation, color="blue")
plt.title('Precipitation vs Number of Bicyclist')
plt.xlabel('Bicyclist')
plt.ylabel('Precipitation')
plt.grid(True)
plt.show()

Y = np.array(precipitation).reshape((-1, 1))
X = total

X1 = stats.add_constant(X)
model = stats.OLS(Y, X1)
results = model.fit()
print("params: \n", results.params)
print("R squared:", results.rsquared)

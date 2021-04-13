#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
import re
import csv
import sys
import pandas as pnd
import numpy as nmp
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data", help="data input as csv file")
parser.add_argument("--learningRate",type=float, help="Learning Rate")
parser.add_argument("--threshold",type=float, help="Threshold")
args = parser.parse_args()





#setting variable
data=args.data
learningRate=args.learningRate
iteration=0
threshold =args.threshold
sum_of_squared_errors = 1
data_out =[]


#reading the data file 
input_data = pnd.read_csv(data, header=None)


#getting the row and column count
rows,cols = input_data.shape


#splitting the data to X and y
X=[]
y=[]
X = input_data.iloc[:,0:cols-1]
y = input_data.iloc[:,cols-1:cols]
y= y.to_numpy()


#adding X0 column with values 1
X0=nmp.ones((rows,1))
X=nmp.hstack((X0,X))




# summation of xi and yi for each independent variable
fvalue=0.0000
output=[]
for i in range(0, cols):
    fvalue_total=0.0000
    for j in range(0, rows):
        y_value=nmp.asscalar(y[j])
        value=X[j][i]
        fvalue= (nmp.asscalar(value)*y_value)
        fvalue_total=(fvalue_total + fvalue)
    output.append(fvalue_total)



# initializing weight to zeto

weight=[0]*cols


#function value calculation
def calc_function(weight, X):
    function = []
    for i in range(0, rows):
        function_value = 0
        for j in range(0, cols):
            value = X[i][j]
            function_value = function_value + X[i][j] * weight[j]
        function.append(function_value)
    return function


# weight calculation for each iteration
def calc_weight(weight, output, eval_function):
    new_weight = []
    for i in range(0, cols):
        gradient = output[i] - eval_function[i]
        new_weight_value = weight[i] + learningRate * gradient
        new_weight.append(round(new_weight_value, 4))
    return new_weight



# Iteration
while threshold != sum_of_squared_errors:
    function = calc_function(weight, X)
    eval_function = []
    for j in range(0, cols):
        eval_function_value=0
        for i in range(0, rows):
            value=X[i][j]
            eval_function_value=eval_function_value + value * function[i]
        eval_function.append(eval_function_value)     
    sum_of_squared_errors_value = sum_of_squared_errors
    sum_of_squared_errors = 0
    for i in range (0, rows):
        error = function[i] - y[i]
        squared_error = error * error
        sum_of_squared_errors = sum_of_squared_errors + squared_error
    sum_of_squared_errors=nmp.asscalar(sum_of_squared_errors)
    sum_of_squared_errors = round(sum_of_squared_errors, 4)

    #print("Iteration", iteration,weight, sum_of_squared_errors)
    #storing values for printing to csv file
    row_out = "%d,%s,%.4f" % (iteration, ', '.join(map(str, weight)), sum_of_squared_errors)
    #print(row_out)
    #print (', '.join(map(str, weight)))
    data_out.append([row_out])
    weight_value = weight
    new_weight = calc_weight(weight, output, eval_function)
    weight = new_weight
    iteration = iteration + 1

    # To avoid infinite loop
    if sum_of_squared_errors_value == sum_of_squared_errors:
        print("Ending the loop")
        break
        
#Generate output csv
file_name = re.sub('\.csv', '', os.path.basename(data))
out_csv = "solution_" + file_name + "_eta" + str(learningRate) + "_thres" + str(threshold) + ".csv"
print("Output stored in current directory,File name:", out_csv)
with open(out_csv, mode='w', newline='') as output:
    output_writer = csv.writer(output)
    for values in data_out:
        output_writer.writerow(values)







    
                



# In[ ]:





# In[ ]:





# In[ ]:





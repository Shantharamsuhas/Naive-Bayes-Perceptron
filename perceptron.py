import numpy as np
import pandas as pnd
import csv
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data", help="data input as tsv file")
parser.add_argument("--output", help="Output file name")
args = parser.parse_args()


#setting variable for argument passing
data=args.data
output_filename=args.output 


#reading the file 
input_data = pnd.read_csv(data,delimiter='\t', header=None)

#getting the row and column count
rows,cols = input_data.shape

#splitting the data to X and y
X=[]
y=[]
X = input_data.iloc[:,1:cols-1]
y = input_data.iloc[:,0]
y = y.replace("A",1).replace("B",0)
y= y.to_numpy()

#adding X0 column with values 1
X0=np.ones((rows,1))
X=np.hstack((X0,X))

# initializing weight to zero
weight=[0]*(cols-1)
annealing_weight=[0]*(cols-1)

# intializing values
error_count=[]
error_count_annealing=[]
epoch=100
iteration=0
learning_rate=1


def activation_function(y):
    y_output=[]
    for i in range(0,rows):
        if y[i]>0:            
            y_output.append(1)
        else:
            y_output.append(0)
    return y_output


def calc_function(weight,X):
    function_value=np.matmul(X,np.asarray(weight))
    return function_value



def calc_error(y,function):    
        error_value= y-np.asarray(function) 
        return error_value

    
# Begin the iteration
print("error calculation started")
while iteration<=(epoch):
    #normal learning rate    
    function_o = calc_function(weight, X)    
    activation_function_o=activation_function(function_o)     
    error_output=calc_error(y,activation_function_o)     
    weight =weight+ learning_rate * np.matmul(np.transpose(X), np.asarray(error_output))    
    error_count.append(np.count_nonzero(error_output))
    
    #with annealing learning rate
    function_o_annealing=calc_function(annealing_weight,X)
    activation_function_o_annealing=activation_function(function_o_annealing)
    error_output_annealing=calc_error(y,activation_function_o_annealing)
    learning_rate_annealing=(learning_rate)/(iteration+1)
    annealing_weight = annealing_weight + learning_rate_annealing * np.matmul(np.transpose(X), np.asarray(error_output_annealing))  
    error_count_annealing.append(np.count_nonzero(error_output_annealing))
    
    #Iteration
    iteration=iteration+1

#Write to file the error count
with open(output_filename,mode='w',newline='') as output:
    print("writing the error output file")
    output_write=csv.writer(output, delimiter='\t')
    output_write.writerow(error_count)
    output_write.writerow(error_count_annealing)
    print("completed file write")








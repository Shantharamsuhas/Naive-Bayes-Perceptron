# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 18:57:05 2019

@author: suhas
"""

import pandas as pd
import numpy as np
import csv
import argparse

parser=argparse.ArgumentParser(description = " Naive Bayes")
parser.add_argument("--data",type =str)
parser.add_argument("--output",type=str)
args=parser.parse_args()
input_filename = args.data
output_filename = args.output
df= pd.read_csv(input_filename,sep='\t',header=None)
#print(df)
dataset=df.dropna(axis=1,how='all')
#print(dataset)
data =dataset.groupby(df[0])
mean=data.mean().to_dict()
print(mean)
variance = data.var().to_dict()
print(variance)
class_prob =(dataset[0].value_counts()/dataset.shape[0]).to_dict()
#print(class_prob)
class_unique = dataset[0].unique().tolist()

print(mean,variance)
for i in dataset.index:
    sample_value=dataset.loc[i].to_dict()
   # print(sample_value)
    ProbA_1= (1/np.sqrt(2*np.pi*variance[1]['A']))*np.exp(-(((sample_value[1]-mean[1]['A']))**2)/(2*variance[1]['A']))
    #print(ProbA_1)
    ProbA_2= (1/np.sqrt(2*np.pi*variance[2]['A']))*np.exp(-(((sample_value[2]-mean[2]['A']))**2)/(2*variance[2]['A']))
    ProbB_1= (1/np.sqrt(2*np.pi*variance[1]['B']))*np.exp(-(((sample_value[1]-mean[1]['B']))**2)/(2*variance[1]['B']))
    ProbB_2= (1/np.sqrt(2*np.pi*variance[2]['B']))*np.exp(-(((sample_value[2]-mean[2]['B']))**2)/(2*variance[2]['B']))
    Prob_A = (ProbA_1*ProbA_2*class_prob['A'])/(ProbA_1*ProbA_2*class_prob['A'])+(ProbB_1*ProbB_2*class_prob['B'])
    Prob_B = (ProbB_1*ProbB_2*class_prob['B'])/(ProbB_1*ProbB_2*class_prob['B'])+(ProbA_1*ProbA_2*class_prob['A'])
    
    if Prob_A >= Prob_B:
        dataset.at[i,'predicted']='A'
    else:
        dataset.at[i,'predicted']='B'
dataset['misclassified']=dataset[0]!=dataset['predicted']
misclass=dataset.groupby(dataset['misclassified']).count()

print((dataset.shape[1]-3))

with open(output_filename,'w') as file :
    for x in class_unique:
        for y in range(1,dataset.shape[1]-2):
            file.write(str(mean[y][x]))
            file.write('\t')
            file.write(str(variance[y][x]))
            file.write('\t')
        file.write(str(class_prob[x]))
        file.write('\n')
    file.write(str(misclass[False][0]))
            
            




              
        
              
             
                     
                     
                     
                     
                     
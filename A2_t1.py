# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
#Usage : "python3 A2_t1.py cm1.txt"
#Argument List: ['A2_t1.py', 'cm1.txt']
python_filename = sys.argv[0]
matrix_filename = sys.argv[1]

def readConfusionMatrix(matrix_filename):
    
    #read confusion matrix text file
    cm_file = np.genfromtxt(matrix_filename, skip_header=1)
    conf_matrix = np.delete(cm_file, 0, 1)
    result = calculatePerformance(conf_matrix)
    
    #get class names(index) from file
    file = open (matrix_filename, 'r')
    arr = []
    arr = [i.split() for i in file]
    indxarr = arr[0]
    
    #print performace matrix
    print("\nAc ", format(result[0], '.2f'))
    data = {'P   ': result[1], 'R   ': result[2], 'Sp  ': result[3], 'FDR ': result[4]}
    df = pd.DataFrame(data = data, index = indxarr)
    print(df)
    return conf_matrix

def calculatePerformance(conf_matrix):
    
    #calculate TP, FP, FN, FN
    TP = np.diag(conf_matrix)
    FP = np.sum(conf_matrix, axis=1) - TP
    FN = np.sum(conf_matrix, axis=0) - TP
    TN = np.sum(conf_matrix) - (FP + FN + TP)

    #calculate performance
    accuracy = np.sum(np.diag(conf_matrix)/np.sum(conf_matrix))
    precision = np.round((TP / (TP + FP)), 2)
    recall = np.round((TP / (TP + FN)), 2)
    specificity = np.round((TN / (FP + TN)), 2)
    FDR =  np.round((FP / (TP + FP)), 2)
    
    return accuracy, precision, recall, specificity, FDR

readConfusionMatrix(matrix_filename)


                          
    






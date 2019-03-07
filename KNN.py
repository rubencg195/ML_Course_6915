# -*- coding: utf-8 -*-
"""KNN From Scratch - Assignment #1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/193b21Wymzl12LnkDd178UDqjcKVb97WO

# KNN From Scratch

**Group Members:**

1. Ruben Chevez
2. Kratika Naskulwar

*Code adapted from the following blog post*

https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#@title Hyperparameters
k_distance = 5 #@param {type:"number"}

traning_filename = 'TrainingData_A1.tsv' #@param {type:"string"}

test_filename = 'TestData_A1.tsv' #@param {type:"string"}

standalone_file = True #@param ["False", "True"] {type:"raw"}

debug = False #@param ["False", "True"] {type:"raw"}

"""# Upload Training and Test data Set"""

if not standalone_file:
  from google.colab import files
  import os
  
  print("\nUpload Traning Set ", traning_filename)
  if not os.path.isfile(traning_filename):
    training_data = files.upload()

  print("\nUpload Test Set ", test_filename)
  if not os.path.isfile(test_filename):
    test_data = files.upload()

else:
  import sys
  #Argument List: ['KNN.py', 'train.tsv', 'test.tsv', 'K']
  python_filename    = sys.argv[0]
  traning_filename   = sys.argv[1]
  test_filename      = sys.argv[2]
  k_distance         = sys.argv[3]
  if debug:
    print("Python Filename: {} \nTraining Data Filename: {}\nTest Data Filename {} \n K: {} ".format( 
      python_filename,
      traning_filename, 
      test_filename, 
      k_distance ))

"""# Importing the Data

This section is to load the training and the test data files using read_csv function.
"""

import pandas as pd
train_data = pd.read_csv(traning_filename, sep='\t')
test_data  = pd.read_csv(test_filename, sep='\t')

if debug: 
  print("Traning Set Head \n\n", train_data.head(5), "\n")
  print("Test Set Head \n\n", test_data.head(5))
  print("\nType of Classes: \n\n", pd.Series(train_data.as_matrix()[:,  -1], name='A').unique() )

"""# Training on Data
Distance - euclideanDistance function is to find k-most similar instances in training data for an instance of test data, we need to find the distance. As all the attributes are numeric , we used Eucleadian Distance calculation which is calulated as the square root of the sum of the squared differences between the two points.

Neighbours - getNeighbors function finds the k-most nearest neighbours from the training dataset for a given test instance.

 trainSet = [[2, 2, 2, 'a'], [4, 4, 4, 'b']]

 testInstance = [5, 5, 5]
"""

import math
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

import operator 
def getNeighbours(trainingInstance, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingInstance)):
		dist = euclideanDistance(testInstance, trainingInstance[x], length)
		distances.append((trainingInstance[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbours = []
	for x in range(int(k)):
		neighbours.append(distances[x][0])
	return neighbours

data1 = train_data.as_matrix(columns=None)[:][0]
data2 = train_data.as_matrix(columns=None)[:][1]
number_of_attributes = 9
distance = euclideanDistance(data1, data2, number_of_attributes)

if debug:
  print("Calculating Distance for single instance")
  print("Data 1 : ", data1, "\n")
  print("Data 2 : ", data2, "\n")
  print( 'Distance: ' + repr(distance) )

trainingInstance = train_data.as_matrix(columns=None)
testInstance = test_data.as_matrix(columns=None)[:][0]
neighbours = getNeighbours(trainingInstance, testInstance, k_distance)

if debug:
  print("Test the getNeighbors function \n\n")
  print("Shape train set:     ", trainingInstance.shape  )
  print("Shape test instance: ", testInstance.shape  )
  print("\nGet The Neighbors for one test instance based on the train set: \n\n")
  print(pd.DataFrame(neighbours, columns=["RI","Na","Mg","Al","Si","K","Ca","Ba","Fe","Class"]))

"""# Prediction based on Majority Vote
getResponse function is  to find the class which has the majority votes among the nearest neighbours.
"""

def getPredictions(neighbours):
  class_votes = {}
  for x in range(len(neighbours)):
    prediction = neighbours[x][-1]
    if prediction in class_votes:
      class_votes[prediction] += 1
    else:
      class_votes[prediction] = 1
  majority_votes = sorted(
    class_votes.items(), 
    key=operator.itemgetter(1), 
    reverse=True
  )
#   print("Majority vote table: \n\n {} \n".format(pd.DataFrame(majority_votes, columns=["Class","Votes"])) )
  return majority_votes[0][0]  , majority_votes[0][1] / len(neighbours) * 100

prediction = getPredictions(neighbours)

if debug: 
  print("{} \n\nBelongs to Class: {} \nWith a confidence of {}".format(
      pd.DataFrame(neighbours, columns=["RI","Na","Mg","Al","Si","K","Ca","Ba","Fe","Class"]), 
      str(int(prediction[0])),
      prediction[1]
  ))

"""#Calculate with all DataSet


testSet    = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]

predictions = ['a', 'a', 'a']
"""

trainingSet  = train_data.as_matrix(columns=None)
testSet      = test_data.as_matrix(columns=None)

predictions=[]
for x in range(len(testSet)):
  neighbors = getNeighbours(trainingSet, testSet[x], k_distance)
  result = getPredictions(neighbors)
  predictions.append(result)
  print('{}\t{}'.format( 
      str(int(result[0])),  
      result[1]
  ))
  
if debug:
  print("\nShape test set:       ",   testSet.shape )
  print("\nShape predictions set: ", len(predictions)  )
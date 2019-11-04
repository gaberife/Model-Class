import pandas as pd
import math
import numpy as np
import scipy.spatial
import timeit
from sklearn import model_selection
from scipy.spatial import distance 
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.scorer import make_scorer

# hw05 Paste 
def readData(numRows = None):
    inputCols = ["Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium", "Total Phenols", "Flavanoids", "Nonflavanoid Phenols", "Proanthocyanins", "Color Intensity", "Hue", "Diluted", "Proline"]
    outputCol = 'Class'
    colNames = [outputCol] + inputCols  # concatenate two lists into one
    wineDF = pd.read_csv("data/wine.data", header=None, names=colNames, nrows = numRows)
    wineDF=wineDF.sample(frac=1, random_state=99).reset_index(drop=True)
    return wineDF, inputCols, outputCol

def foldsTest(trainInputDF, trainOutputSeries, testInputDF, testOutputSeries):
    print("================================")
    print("Train input:\n", list(trainInputDF.index))  # convert to list to print all contents
    print("Train output:\n", list(trainOutputSeries.index))  # convert to list to print all contents
    
    print("Test input:\n", testInputDF.index)
    print("Test output:\n", testOutputSeries.index)

    return 0

def kFoldCVManual (k, inputDF, outputSeries, func):
    
    size = inputDF.shape[0]  #finds size of the inputDF
    foldSize = size / k  #determines the size of each fold 
    
    results = 0 #init array to store results of folds test
   
    for i in range(k):
       start = int(i*foldSize) 
       upToNotIncluding = int((i+1)*foldSize)
       #determins the boundaries of each fold left off from the last iteration
       testInputDF = inputDF.iloc[start:upToNotIncluding] #the input columns for the testing set
       upperSeg = inputDF.iloc[:start,:]
       lowerSeg = inputDF.iloc[upToNotIncluding:,:]
       trainInputDF = pd.concat([upperSeg,lowerSeg]) #the input columns for the training set
       testOutputSeries = outputSeries.iloc[start:upToNotIncluding] #the output column for the testing set
       upperSeries = outputSeries.iloc[:start]
       lowerSeries = outputSeries.iloc[upToNotIncluding:]
       trainOutputSeries = pd.concat([upperSeries,lowerSeries]) #the output column for the training set
       results += func(trainInputDF, trainOutputSeries, testInputDF, testOutputSeries)
    return results / k

# ---------------------------------
# Problem 2
def partialOneNNTest():
    df, inputCols, outputCol = readData()
    # Just arbitrarily choose a small training and testing set from the entire df, for easy testing
    # Note how I'm chaining a .loc with a .iloc so I can use the indexers I want for row and column
    trainInputDF      = df.loc[:, inputCols].iloc[125:131, :]
    trainOutputSeries = df.loc[:, outputCol].iloc[125:131]
    testInputDF       = df.loc[:, inputCols].iloc[131:135, :]
    testOutputSeries  = df.loc[:, outputCol].iloc[131:135]
    
    return oneNNTest(trainInputDF, trainOutputSeries, testInputDF, testOutputSeries)

def findNearestHOF(df,testRow):
    nearestHOF = df.apply(lambda row: distance.euclidean(row, testRow),axis = 1)
    return nearestHOF.idxmin()  


def oneNNTest(trainInputDF, trainOutputSeries, testInputDF, testOutputSeries):
    firstSeries = testInputDF.apply(lambda r: findNearestHOF(trainInputDF, r),axis = 1) #series from DF of nearest indexes
    secondSeries = firstSeries.map(lambda r: trainOutputSeries.loc[r]) #series fo predicted 
    return accuracyOfActualVsPredicted(testOutputSeries,secondSeries)
    
# Given
def accuracyOfActualVsPredicted(actualOutputSeries, predOutputSeries):
    compare = (actualOutputSeries == predOutputSeries).value_counts()
    
    # actualOutputSeries == predOutputSeries makes a Series of Boolean values.
    # So in this case, value_counts() makes a Series with just two elements:
    # - with index "False" is the number of times False appears in the Series
    # - with index "True" is the number of times True appears in the Series

    # print("compare:", compare, type(compare), sep='\n', end='\n\n')
    
    # if there are no Trues in compare, then compare[True] throws an error. So we have to check:
    if (True in compare):
        accuracy = compare[True] / actualOutputSeries.size
    else:
        accuracy = 0
    
    return accuracy

# --------------------------------------------------------------------------------------
# Problems 4-6
    
class OneNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.inputsDF = None
        self.outputSeries = None
        self.scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True)    
    def fit(self, inputsDF, outputSeries):
        self.inputsDF = inputsDF
        self.outputSeries = outputSeries
        return self
    def predict(self, testInput):
        if isinstance(testInput, pd.core.series.Series):
            return self.outputSeries.loc[ findNearestHOF(self.inputsDF, testInput)]
        else:
            series = testInput.apply(lambda r: findNearestHOF(self.inputsDF, r),axis = 1)
            newSeries = series.map(lambda r: self.outputSeries.loc[r])
            return newSeries

# Given
def testTheClass():
    fullDF, inputCols, outputCol = readData()
    
    # Use rows 0:100 for the training set
    trainDF = fullDF.iloc[0:100, :]
    
    # Set up the inputs and outputs for the training set
    trainInputDF = trainDF.loc[:, inputCols]
    trainOutputSeries = trainDF.loc[:, outputCol]
    
    # Make the classifier object and "fit" the training set
    alg = OneNNClassifier()
    alg.fit(trainInputDF, trainOutputSeries)
    
    # Use rows 100:200 for the testing set
    testDF = fullDF.iloc[100:200, :]
    
    # Set up the inputs for the testing set
    testInputDF = testDF.loc[:, inputCols]
    
    # Predict outputs for just a single row (a Series)
    print("Series:", alg.predict(testInputDF.iloc[0]), end='\n\n')
    
    # Predict outputs for the entire testing set (a DataFrame)
    print("DF:", alg.predict(testInputDF), sep='\n')

# --------------------------------------------------------------------------------------
# Problem 9

# Given
def testBuiltIn():
    fullDF, inputCols, outputCol = readData()
    result = kFoldCVBuiltIn(3, fullDF.loc[:, inputCols], fullDF.loc[:, outputCol])
    print(result)

# Given
def compareManualAndBuiltIn(k=10):
    df, inputCols, outputCol = readData()
    
    inputDF = df.loc[:, inputCols]
    outputSeries = df.loc[:, outputCol]
    
    results = kFoldCVBuiltIn(k, inputDF, outputSeries)
    print("Built-in:", results)
    
    results = kFoldCVManual(k, inputDF, outputSeries, oneNNTest)
    print("Manual:", results)

def kFoldCVBuiltIn(k,inputDF,outputSeries):
    alg = OneNNClassifier()
    cvScores = model_selection.cross_val_score(alg, inputDF, outputSeries, cv=k, scoring=alg.scorer)
    return cvScores.mean()

# --------------------------------------------------------------------------------------
# Given
def test06():
    df, inputCols, outputCol = readData()
    alg = OneNNClassifier()
    alg.fit(df.loc[:, inputCols], df.loc[:, outputCol])
    print(model_selection.cross_val_score(alg, df.loc[:, inputCols], df.loc[:, outputCol], cv=10, scoring=alg.scorer).mean())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 08:34:35 2018

@author: abhishek
"""
import nltk
posFilePath='./rt-polaritydata/rt-polaritydata/rt-polarity.pos'
negFilePath='./rt-polaritydata/rt-polaritydata/rt-polarity.neg'
#print(sia['compoud'])/home/abhishek/ML/sentiment analysis/rt-polaritydata/rt-polaritydata

with open(posFilePath, encoding="latin-1") as f:
    posData=f.readlines()
    
with open(negFilePath,encoding="latin-1") as f:
    negData=f.readlines()

testSpltIndx=2500
testNegRev=negData[testSpltIndx+1:]
testPosRev=posData[testSpltIndx+1:]

trainNegRev=negData[:testSpltIndx]
trainPosRev=posData[:testSpltIndx]

def getVocab():
    posWordLst = [word for line in trainPosRev for word in line.split()]
    negWordLst = [word for line in trainNegRev for word in line.split()]
    allWordsList = posWordLst+negWordLst
    vocab = list(set(allWordsList))
    
    return vocab

def getTrainingData():
    negTaggedRevLst = [{'review':review.split(),'label':'negative'} for review in trainNegRev]
    posTaggedRevLst = [{'review':review.split(),'label':'positive'} for review in trainPosRev]
    fullyTaggedTrainData = negTaggedRevLst+posTaggedRevLst

    trainData =[(reviewObj['review'],reviewObj['label']) for reviewObj in           fullyTaggedTrainData]
    return trainData


def extractFeatures(review):
    review_words=set(review)
    features={}
    for word in vocabulary:
        features[word]=(word in review_words)
    
    return features

    
def getTrainedNaiveBayesClassifier(extract_feaures,trainingData):
    trainingFeatures = nltk.classify.apply_features(extract_feaures,trainingData)
    trainedNBClassifier = nltk.NaiveBayesClassifier.train(trainingFeatures)
    return trainedNBClassifier    

vocabulary = getVocab()
trainingData = getTrainingData()

trainedNBClassifier = getTrainedNaiveBayesClassifier(extractFeatures,trainingData)


def nbSentimentCalc(review):
    problemInstance = review.split()
    problemFeatures = extractFeatures(problemInstance)
    return trainedNBClassifier.classify(problemFeatures)

nbSentimentCalc("what an bad good movie")




    
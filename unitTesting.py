
from sklearn import tree
import pandas as pd
import numpy as np
import itertools as itt
from sklearn.neighbors import KNeighborsClassifier

# Testing to make sure tree classifier behaves properly (lower class)
def test_tree1():
    junk = [{'x':1,'y':1},{'x':2,'y':1},{'x':1,'y':2},{'x':10,'y':50},{'x':12,'y':52},{'x':11,'y':51}]
    junkLab = [{'z':1},{'z':1},{'z':1},{'z':2},{'z':2},{'z':2}]
    junkT = [{'x':0,'y':0}]
    junkTLab = [{'z':1}]
    trainDat = pd.DataFrame(junk)
    trainLab = pd.DataFrame(junkLab)
    testDat = pd.DataFrame(junkT)
    testLab = pd.DataFrame(junkTLab)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(trainDat,trainLab)
    assert clf.predict(testDat) == 1, clf.predict(testDat)
    
    
# Testing to make sure tree classifier behaves properly (higher class)
def test_tree2():
    junk = [{'x':1,'y':1},{'x':2,'y':1},{'x':1,'y':2},{'x':10,'y':50},{'x':12,'y':52},{'x':11,'y':51}]
    junkLab = [{'z':1},{'z':1},{'z':1},{'z':2},{'z':2},{'z':2}]
    junkT = [{'x':100,'y':80}]
    junkTLab = [{'z':1}]
    trainDat = pd.DataFrame(junk)
    trainLab = pd.DataFrame(junkLab)
    testDat = pd.DataFrame(junkT)
    testLab = pd.DataFrame(junkTLab)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(trainDat,trainLab)
    assert clf.predict(testDat) == 2, clf.predict(testDat)
    
    
# Testing to make sure we are getting only a certain subset of combinations
def test_combo():
    leage = [1,2,3,4,5,6]
    testA = [[1,2],[5,6],[2,4],[3,6]]
    fin = []
    for a in itt.combinations(leage,2):
         if ((a[0]==1)&(a[1]==2))|((a[0]==5)&(a[1]==6))|((a[0]==2)&(a[1]==4))|((a[0]==3)&(a[1]==6)):
                 fin.append([a[0],a[1]])
                    
        
    assert all(g in testA for g in fin),g
    
    
# Testing to make sure KNN behaves properly (lower class)
def test_knn():
    junk= [{'x':1,'y':1},{'x':2,'y':1},{'x':1,'y':2},{'x':10,'y':50},{'x':12,'y':52},{'x':11,'y':51}]
    junkLab= [{'z':1},{'z':1},{'z':1},{'z':2},{'z':2},{'z':2}]
    junkT= [{'x':0,'y':0}]
    junkTLab=[{'z':1}]
    trainDat=pd.DataFrame(junk)
    trainLab=pd.DataFrame(junkLab)
    testDat=pd.DataFrame(junkT)
    testLab=pd.DataFrame(junkTLab)
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(trainDat, np.ravel(trainLab)) 
    predictedValues = KNeighborsClassifier.predict(neigh,testDat)
    assert predictedValues ==1, predictedValues
    

# Testing to make sure KNN behaves properly (higher class)
def test_knn2():
    junk= [{'x':1,'y':1},{'x':2,'y':1},{'x':1,'y':2},{'x':10,'y':50},{'x':12,'y':52},{'x':11,'y':51}]
    junkLab= [{'z':1},{'z':1},{'z':1},{'z':2},{'z':2},{'z':2}]
    junkT= [{'x':15,'y':60}]
    junkTLab=[{'z':2}]
    trainDat=pd.DataFrame(junk)
    trainLab=pd.DataFrame(junkLab)
    testDat=pd.DataFrame(junkT)
    testLab=pd.DataFrame(junkTLab)
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(trainDat, np.ravel(trainLab)) 
    predictedValues = KNeighborsClassifier.predict(neigh,testDat)
    assert predictedValues ==2, predictedValues
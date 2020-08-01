#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import copy
import math
import random
from numpy.linalg import norm
pd.options.mode.chained_assignment = None 
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(0)


# In[5]:


class kmeans():
    
    def __init__(self,data,maxIterations):
        self.trainingset=data
        self.maxIterations=maxIterations
        
    def dataPreprocess(self):
        """The kmeansDataAttributes has the features of all data points and the
        kmeansDatalabels has labels and indices of the respective data points. These two data sets remain untouched"""
        kmeansData=self.trainingset
        kmeansDataAttributes=kmeansData.drop(columns=["Index","Class"])
        kmeansDataLabels=kmeansData[["Index","Class"]]
        return kmeansDataAttributes.values,kmeansDataLabels.values
    
    def getK(self):
        """The getk function returns a k value taken as input from user"""
        print("Enter a K value")
        k=int(input())
        return k
    
    def startingClusters(self):
        """This function take a random k clusters from given N data points and is stored in initialCluster,initialClusterLabel"""
        N=len(self.trainingset)
        index=np.random.randint(0,N,size=self.K)
        initialCluster=[]
        initialClusterLabel=[]
        for ind in index:
            initialCluster.append(self.kmeansDataAttributes[ind])
            initialClusterLabel.append(self.kmeansDataLabels[ind])
        return np.array(initialCluster),np.array(initialClusterLabel)
    
    def eucledian(self):
        distance=[]
        for i in range(len(self.clusterCentroids)):
            distance.append(np.sqrt (np.sum(( np.square ( self.clusterCentroids[i]- self.kmeansDataAttributes) ),axis=1)))
        return np.array(distance)
    
    def clusters(self):
        """This function assigns clusters to all data points based on the eucledian distance calculated"""
        distance=self.eucledian()
        self.cluster=distance.T.argmin(axis=1)
               
    def updateCentroid(self):
        """This function returns the updated centroid used to update the initial centroids after the clustering is done"""
        newMean=[]
        unique=list(np.unique(self.cluster))
        for ind in unique:
            index=np.flatnonzero(self.cluster==ind)
            newMean.append(np.mean(self.kmeansDataAttributes[index],axis=0))
        return np.array(newMean)
    
    def main(self):
        self.kmeansDataAttributes,self.kmeansDataLabels=self.dataPreprocess()
        self.K=self.getK()
        initialCluster,initialClusterLabel=self.startingClusters()
        self.clusterCentroids=initialCluster
        
        iterVar=0
        while iterVar<self.maxIterations:
            self.clusters()
            self.clusterCentroids=self.updateCentroid()
            iterVar+=1    
            
    def dataFrame(self):
        """Contains the newDataframe along with the classified cluster groups"""
        self.result=copy.deepcopy(self.trainingset)
        self.result["clusters"]=self.cluster
        
    def wcSSE(self):
        """Used to calculate the within cluster sum of squares"""
        sse=0
        ctrd=self.clusterCentroids#A temporary variable holding last updated centroids
        clstr=self.cluster#A temporary variable holding last updated clusters
        unique=list(np.unique(clstr))
        for i in unique:
            index=np.flatnonzero(clstr==i)
            temp=self.kmeansDataAttributes[index]#This temp var holds the data for each clusters in respective loops
            sse+=np.sum((np.sum(( np.square ( ctrd[i]- temp) ),axis=1)),axis=0)
        
        return sse
        
    def silhouetteCoeff(self):
        ctrd=self.clusterCentroids#A temporary variable holding last updated centroids
        clstr=self.cluster#A temporary variable holding last updated clusters
        unique=list(np.unique(clstr))
        temp=[]
        
        for i in unique:
            index=np.flatnonzero(clstr==i)
            temp.append(self.kmeansDataAttributes[index])#This temp var holds the data for each clusters in respective loops
        temp=np.array(temp)
        
        otherCluster=len(temp)-1#The  length of other clusters
        
        #Avoid zero division error
        if otherCluster==0:
            otherCluster=1
        
        silCoeff=0
            
        for i in range(len(temp)):
            for row in range(len(temp[i])):
                A=0
                B=0
                for j in range(len(temp)):
                    if i==j:
                        A+=1
                        A+=np.average( norm( temp[i][row] - temp[j] ,axis=1 ) )
                    else:
                        B+=1
                        B+=np.average( norm( temp[i][row] - temp[j],axis=1 ) )     
                B=B/otherCluster
                silCoeff+=((B-A)/max(A,B))
        
        silCoeff=silCoeff/len(self.trainingset)
                
        return silCoeff
    
    def nmi(self):
        nmiData=self.result
        
        #Calc entropy of class labels
        entropyClass=0
        uniqueClass=np.unique(nmiData["Class"])
        classProb={}
        
        for i in uniqueClass:
            classProb[i]=(np.sum(nmiData["Class"]==i))/len(nmiData)
                    
        for value in list(classProb.values()):
            if value==0: 
                entropyClass+=0
            else:
                entropyClass+=(-value*np.log2(value))
                
        #Calc entropy of cluster labels
        entropyCluster=0
        uniqueCluster=np.unique(nmiData["clusters"])
        clusterProb={}
        for j in uniqueCluster:
            clusterProb[j]=(np.sum(nmiData["clusters"]==j))/len(nmiData)          
        for val in list(clusterProb.values()):
            if val==0: 
                entropyCluster+=0
            else:
                entropyCluster+=(-val*np.log2(val))
                
        #Calc conditional entropy
        condEntropy=[]
        clusGroup=[]#A list to hold the cluster groups dataframe
        uniqueCond=np.unique(nmiData["clusters"])       
        for k in uniqueCond:
            clusGroup.append(nmiData[nmiData["clusters"]==k])            
        for idx in range(len(clusGroup)):
            ent=0
            uniqueGrp=np.unique(clusGroup[idx]["Class"]) 
            for vlue in uniqueGrp:
                counts=np.sum(clusGroup[idx]["Class"]==vlue)/len(clusGroup[idx])
                if counts==0:
                    ent+=0
                else:
                    ent+=(counts*np.log2(counts))
            condEntropy.append(ent)    
        
        #Calculate Info gain
        multiplier=list(clusterProb.values())#It is the values of cluster probability
        infoGain=0
        for prb in range(len(condEntropy)):
            infoGain+=-multiplier[prb] * condEntropy[prb]
        infoGain=entropyClass-infoGain   
        
        normMI=infoGain/(entropyClass+entropyCluster)
        
        return normMI
                             


# In[6]:


if __name__ == "__main__":
    trainingSet=pd.read_csv("{}".format(input("Enter the Dataset with csv extension : eg.trainingSet.csv ")),delimiter=',',header=None)
    trainingSet.columns=["Index","Class","X","Y"]
    km=kmeans(trainingSet,50)
    km.main()
    km.dataFrame()
    se=km.wcSSE()
    sc=km.silhouetteCoeff()
    nmi=km.nmi()
    print("WC-SSD : {}".format(se))
    print("SC : {}".format(sc))
    print("NMI : {}".format(nmi))
    


#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Machine Learning W4771, Spring 2019
#Dhruv Sing - ds3721, Kevin Xiong - kjx2000, Richard Pan - rp2876

import sklearn.datasets as skdat
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import itertools as it


# In[2]:


#finds the means of the coordinates
def find_means(clusters):
    means = []
    for i in range(0,len(clusters)):
        means.append(np.mean(clusters[i],axis=0))
    return means


# In[3]:


#checks for convergence
def not_converged(cent, old_cent, count):
    #terminates the program at 1000 iterations
    if (count > 1000):
        return False
    
    if (old_cent == []):
        return True
    
    if (np.array_equal(cent, old_cent)):
        return False
    
    return True


# In[4]:


#determines nearest central point
def nearest(coord, cent):
    min = 1000000.0
    mindex = -1
    
    for i in range(len(cent)):
        dist = np.linalg.norm(coord-cent[i])
        if (dist < min):
            min = dist
            mindex = i
    return mindex


# In[5]:


#lloyd's method in 2-d space
def kmeans(data, k):

    #cent is collection of central points
    #oldcent keeps track of previous iteration in order to check for convergence
    cent = []
    old_cent = []
    labels = []
    
    #initialize random central points
    for i in range(0,k):
        cent.append([rand.uniform(-1,1) for i in range(len(data[0]))])
        
    #count ensures the loop doesnt interate infinitely
    count = 0
    while (not_converged(cent, old_cent, count)):
        count += 1
        
        #initialize the clusters
        clusters = [[] for a in range(k)]
        
        #add coordinates of each data point to the jth cluster depending on which central point it is closest to
        for coord in data:
            i = nearest(coord,cent)
            clusters[i].append(coord)
        
        old_cent = cent
        cent = find_means(clusters)
    
    for i in range (0, len(data)):
        labels.append(nearest(data[i],cent))
        
    return labels
        


# In[6]:


#transformation of data for more flexible clustering
def transformation(data, r, k):
    n = len(data)
    W = np.zeros((n,n))
    D = np.zeros((n,n))
    L = np.zeros((n,n))
    V = np.zeros((n,k))
    
    #creates W matrix
    for i in range(0, len(data)):
        near_index = [-1 for sth in range(r)]
        near_dist = [1000000.0 for sth in range(r)]
        
        #checks for r closest neighbors
        for j in it.chain(range(0,i), range(i+1, len(data))):
            dist = np.linalg.norm(data[j]-data[i])
            
            for a in range(r):
                if (dist < near_dist[a]):
                    near_dist[a] = dist
                    near_index[a] = j
                    break
        
        for a in range(r):
            W[i][near_index[a]] = 1
            W[near_index[a]][i] = 1
    
    #creates D matrix
    for i in range(n):
        row_sum = 0
        for j in range(n):
            row_sum += W[i][j]
        D[i][i] = row_sum
    
    
    #computes L matrix and its eigenvectors / eigenvalues
    L = np.subtract(D,W)
    eigenval, eigenvect = np.linalg.eigh(L)
    

    #creates V matrix
    for i in range(k):
        V[:,i] = eigenvect[:,i]
    
    return V


# In[7]:


#testing kmeans on transformed data
X, Y = skdat.make_circles(n_samples = 100)
#X, Y = skdat.make_blobs(n_samples=np.array([100, 50, 20]), center_box=(-10.0,10.0))
new_data = transformation(X,50,2)
labels = kmeans(new_data, 2)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=labels, edgecolor='k')


# In[8]:


#test on concentric circles
X, Y = skdat.make_circles()
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, edgecolor='k')
plt.show()

print (len(X[1]))
klabels = kmeans(X, 2)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=klabels, edgecolor='k')
plt.show()


# In[9]:


#test on moon shaped data
X, Y = skdat.make_moons()
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, edgecolor='k')
plt.show()

klabels = kmeans(X, 2)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=klabels, edgecolor='k')
plt.show()


# In[10]:


#test on gaussian blobs
X, Y = skdat.make_blobs(n_samples=np.array([100, 50, 20]), center_box=(-10.0,10.0))

plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, edgecolor='k')
plt.show()

klabels = kmeans(X, 3)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=klabels, edgecolor='k')
plt.show()


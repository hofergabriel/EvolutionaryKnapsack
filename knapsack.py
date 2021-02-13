"""
Author: Gabriel Hofer
Date: 01/08/2021
Course: CSC-449
"""
import numpy as np
import random
import time
import matplotlib.pyplot as plt

class EvolutionaryKnapsack:

  def __init__(self,n,psz):
    ## For plotting population over time T
    self.X = []
    self.Y = []
    self.T = 0

    ## initialize boolean object vector, size of population, weight sum 
    self.N = n 
    self.Psize = psz; 
    self.W = 0

    ## Keep track of best 0-1 vector
    self.mx_vector=[]
    self.mx_value=0

    ## Initialize population (of knapsacks) to 8 random vectors
    self.P = [ np.random.randint(2, size=self.N) for j in range(self.N) ]
    self.weights_values = np.random.randint(self.N, size=(self.N,2))

    ## Avoid values of zero in our data
    for i in range(self.weights_values.shape[0]):
      self.weights_values[i,0]+=1 
      self.weights_values[i,1]+=1 

    ## set max weight of knapsack to half of all of the weights
    for i in range(self.weights_values.shape[0]):
      self.W += self.weights_values[i,0] 
    self.W //= 2  

  ## Print maximum weight allowed for a valid Knapsack
  def show_W(self): 
    print("Maximum Knapsack Weight Allowed: "+str(self.W))
  
  ## Print weight-value pairs
  def show_weights_values(self):
    print("Weight-Value Pairs: ")
    print(self.weights_values,end="\n\n")

  ## Print Population
  def show_P(self):
    for e in self.P:
      print(self.show_wv(e),end=' ')
    print()

  ## Print the total weight and total value of a Knapsack
  def show_wv(self,x):
    w=0 ; v=0
    for i in range(len(x)):
      w+=x[i]*self.weights_values[i,0]
      v+=x[i]*self.weights_values[i,1]
    return [w,v]

  ## Returns total value of an individual knapsack
  def getValue(self,x):
    v=0 
    for i in range(len(x)):
      v+=x[i]*self.weights_values[i,1]
    return v

  ## Returns total weights in an individual knapsack
  def getWeight(self,x):
    w=0 
    for i in range(len(x)):
      w+=x[i]*self.weights_values[i,0]
    return w

  ## Returns True if knapsack is fit
  def isFit(self,x): 
    return (self.getWeight(x)<=self.W)

  ## Returns knapsack after adding or removing an object
  def mutate(self,x): 
    x[random.randint(0,x.shape[0]-1)]^=1
    return x

  ## Returns new knapsack after concatenating two halves
  def recombine(self,v0,v1): 
    return np.concatenate((v0[:len(v0)//2],v1[len(v1)//2:]),axis=None)
 
  ## Returns the rank of a knapsack
  def rank(self,x):
    return self.getValue(x)

  ## Selects most fit knapsacks based on rank
  def select(self):
    ret = list(filter(self.isFit,self.P))
    ret.sort(key=self.rank,reverse=True)
    self.P=ret[:self.Psize]
  
  ## Reproduction, Mutation, Recombination, Selection, Keep the best 
  def evolve(self,t): 

    ## Evolve t times
    for i in range(t):

      ## Reproduction and Mutation (3 new children)
      Q=[]
      for j in self.P: 
        Q.append(self.mutate(j))
        Q.append(self.mutate(j))
        Q.append(self.mutate(j))

      ## Recombination
      self.P=[]
      for j in range(len(Q)-1):
        self.P.append(self.recombine(Q[j],Q[j+1]))

      # Selection
      self.select()         

      ## Update best solution 
      if len(self.P)>0 and self.getValue(self.P[0])>self.mx_value:
        self.mx_value=self.getValue(self.P[0])
        self.mx_vector=self.P[0]
        print("mx_value: "+str(self.mx_value))

      ## Add points to the coordinate lists for plotting 
      self.X.append(self.T)
      self.Y.append(self.getValue(self.P[0]))
      self.T += 1

  ## Print the Knapsack with highest Value in population P
  def showBestInP(self):
    print("Evolved Knapsack:    "+str(self.mx_vector))
    print("Evolved Value:       "+str(self.mx_value))

  ## Brute Force Answer - O(2^self.N)
  def showAns(self):
    mx_val = 0
    for i in range(1<<self.N):
      s = bin(i)[2:].zfill(self.N)
      arr = np.zeros(self.N)
      for j in range(len(s)):
        arr[j] = int(s[j])
      w,v=0,0
      for j in range(self.N):
        w+=arr[j]*self.weights_values[j,0]
        v+=arr[j]*self.weights_values[j,1]
      if (w<=self.W) and (v>mx_val):
        mx_arr = arr
        mx_val = v
    print("Answer Knapsack:     "+str(mx_arr.astype(int)))
    print("Answer Value:        "+str(int(mx_val)))

  ## Show Evolutionary Process 
  def showplot(self):
    plt.scatter(self.X,self.Y)
    plt.plot(self.X,self.Y)
    plt.show()
  
## Demo 
def main(): 
  gk = EvolutionaryKnapsack(int(input()),int(input()))
  gk.show_W()
  gk.show_weights_values()

  for i in range(int(input())):
    gk.evolve(1)
    #gk.show_P()

  gk.showBestInP()
  #gk.showAns()
  gk.showplot()

main()







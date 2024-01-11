# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:01:54 2024

@author: Sam Mister
"""
import qutip as qu
import numpy as np
#The primary bit of information needed for the construction of this model is the 
#number of spins in the system.

#Now this following function can be called to generate the pauli spin operators 

def PauliN(N): #define SigmaX and SigmaZ for N spin system
    Sx = [qu.tensor([qu.sigmax() if m == n else qu.identity(2) for n in range(N)]) for m in range(N)]
    Sz = [qu.tensor([qu.sigmaz() if m == n else qu.identity(2) for n in range(N)]) for m in range(N)]
    Sy = [qu.tensor([qu.sigmay() if m == n else qu.identity(2) for n in range(N)]) for m in range(N)]
    return Sx, Sz, Sy

# Next the disordered parameters are needed to build the hamiltonian

def disorderGen(Jmean,Jvar,hmean,hvar,N): # Generates the unifrom distributions for sampling the parameters
    Jarr = [Jmean + np.random.uniform(-Jvar/2,Jvar/2) for i in range(N-1)]
    harr =[hmean + np.random.uniform(-hvar/2,hvar/2) for i in range(N)]  #[np.pi for i in range(N)] 
    return Jarr, harr 

#The system Hamiltonions can now be generated
#This also includes the option to add an external field.  
def HamiltonianGenerator(Jarr, harr, ExternalFieldStength, Sx, Sz): #Constructs the system Hamiltonions 
    
    N = len(Sx)

    HzTerm1 = qu.Qobj()
    for i in range(N):
        HzTerm1 += 0.5*harr[i]*Sz[i]
    
    HzTerm2 = qu.Qobj()
    HxTerm1 =  qu.Qobj()
    for i in range(N-1):
        HzTerm2 += ExternalFieldStength*(Sz[i]*Sz[i+1])
        HxTerm1 += Jarr[i]*(Sx[i]*Sx[i+1])#
    
    Hz = HzTerm1 + HzTerm2
    Hx = HxTerm1 + HzTerm2

    return Hz,Hx 
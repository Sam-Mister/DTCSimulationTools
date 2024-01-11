# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:32:12 2024

@author: sm15883
"""
import numpy as np
import qutip as qu
from SystemToolbox import PauliN
from SystemToolbox import disorderGen
from SystemToolbox import HamiltonianGenerator
from SystemToolbox import GeneralGlobalME 
import matplotlib.pyplot as plt
#Parameters for the closed system
NumberOfSpins = 2
Jmean = np.pi/2
Jvar = 0
hmean = np.pi
hvar = 0.5
ExternalFieldStength = 0

#Bath parameters 
s = 1 #This corresponds to an ohmic bath 
G = 1 # This parameter represents the system bath coupling strength
B = 1 # This is the inverse temperature of the bath 

Sx, Sz, Sy = PauliN(NumberOfSpins)

Jarr, harr = disorderGen(Jmean,Jvar,hmean,hvar,NumberOfSpins)

Hz,Hx = HamiltonianGenerator(Jarr, harr, ExternalFieldStength, Sx, Sz)

'''
We assume that each spin of the model couples to the bath equally 
'''
VAll = qu.Qobj() #defines the operator through with the system couples to the enviroment. 
for i in range(len(Sz)):
    VAll += Sz[i]

LX,LZ = GeneralGlobalME(Hx,Hz,s,G,B,VAll)

'''
At this point we have the components require to build the master equations
needed to evolve the system. Thus I construct a function that takes these objects,
with a few additional things, and returns the time evolution of the state.
'''
def SystemOperators():
    return Hz,Hx,LZ,LX

def TimeEvolver(Np,TimeStep):
    Tz = 1 #time for z evolution
    Tx = 1 #time for x evolution
    
    TotalTime = np.arange(0,Np*(Tz+Tx),TimeStep)
    TzTime = np.arange(0,Tz,TimeStep)
    TxTime = np.arange(0,Tx,TimeStep)
    #psi0Therm = (-B*Hz).expm()/((-B*Hz).expm()).tr()
    psi0 = qu.tensor([qu.Qobj([[1/np.sqrt(2)],[1/np.sqrt(2)]]) for n in range(NumberOfSpins)])
    States = []
    for i in range(Np):
        output = qu.mesolve(Hz, psi0, TzTime,LZ,[])# Evolve for the Hz period 
        States += output.states 
        output1 = qu.mesolve(Hx, output.states[-1], TxTime,LX,[])# Evolve for the Hx period 
        States += output1.states 
        psi0 = output1.states[-1]
    return TotalTime, States
if __name__ == "__main__":
    Time, Results = TimeEvolver(5,0.1)
    
    SxSignature = []
    for i in range(len(Results)):
        SxSignature.append((Results[i]*Sx[0]).tr().real)
        
    plt.plot(Time,SxSignature)
    plt.show()
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

'''
At this point we have the components require to build the master equations
needed to evolve the system. Thus I construct a function that takes these objects,
with a few additional things, and returns the time evolution of the state.
'''
def SystemOperators(params):
    #params = {"NumSpins":5, "Jmean":1, "Jvar":0.5, "hmean":3,"hvar":0,ExternalField":0, "s":1,"G":0.1,"B":1}

    Sx, Sz, Sy = PauliN(params["NumSpins"])

    Jarr, harr = disorderGen(params["Jmean"],params["Jvar"],params["hmean"],params["hvar"],params["NumSpins"],params["seed"])

    Hz,Hx = HamiltonianGenerator(Jarr, harr, params["ExternalField"], Sx, Sz)
    
    VAll = qu.Qobj() #defines the operator through with the system couples to the enviroment. 
    for i in range(len(Sz)):
        VAll += Sz[i]
    
    LX,LZ = GeneralGlobalME(Hx,Hz,params["s"],params["G"],params["B"],VAll)
    return Hz,Hx,LZ,LX,Sx, Sz, Sy

def TimeEvolver(Np,TimeStep,psi0,params):
    Hz,Hx,LZ,LX,Sx, Sz, Sy = SystemOperators(params)
    T = params["Tz"] + params["Tx"]
    
    TotalTime = np.arange(0,Np*(T),TimeStep)
    TzTime = np.arange(0,params["Tz"],TimeStep)
    TxTime = np.arange(0,params["Tx"],TimeStep)
    
    States = []
    for i in range(Np):
        output = qu.mesolve(Hz, psi0, TzTime,LZ,[])# Evolve for the Hz period 
        States += output.states 
        output1 = qu.mesolve(Hx, output.states[-1], TxTime,LX,[])# Evolve for the Hx period 
        States += output1.states 
        psi0 = output1.states[-1]
    return TotalTime, States
if __name__ == "__main__":
    #Parameters for the closed system
    NumberOfSpins = 2
    Jmean = np.pi/2
    Jvar = 0.5
    hmean = np.pi
    hvar = 0.01
    ExternalFieldStength = 0

    #Bath parameters 
    s = 1 #This corresponds to an ohmic bath 
    G = 0.1 # This parameter represents the system bath coupling strength
    B = 1 # This is the inverse temperature of the bath 
    params = {"NumSpins":NumberOfSpins, "Jmean":Jmean, "Jvar":Jvar, "hmean":hmean,"hvar":hvar,"ExternalField":ExternalFieldStength,"Tz":1,"Tx":1,"s":s,"G":G,"B":B}
    
    Hz,Hx,LZ,LX,Sx, Sz, Sy = SystemOperators(params)
    
    Time, Results = TimeEvolver(5,0.01,params)
    
    SxSignature = []
    for i in range(len(Results)):
        SxSignature.append((Results[i]*Sx[0]).tr().real)
        
    plt.plot(Time,SxSignature)
    plt.show()
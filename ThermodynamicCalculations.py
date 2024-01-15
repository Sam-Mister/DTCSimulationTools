# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 17:31:40 2024

@author: sm15883
"""

'''
Given a set of system parameters this script uses functions from the TimeEvolver 
Module to generate the time evolved state rho(t), this combined with the Hamiltonians
and the dissipation operators gives us the required information to calculate the 
thermodynamic properties 
'''

'''
The required parameters from the highest level of simulation will be provided in a 
dictionary and passed along. Closed system parameters listed first, then bath parameters
For example considering the following
'''
params = {"NumSpins":5, "Jmean":1, "Jvar":0.5, "hmean":3,"hvar":0, "ExternalField":0,"Tz":1,"Tx":1, "s":1,"G":0.1,"B":1}

from TimeEvolver import TimeEvolver
from TimeEvolver import SystemOperators
from scipy import linalg
import qutip as qu
import matplotlib.pyplot as plt
import numpy as np

'''
Now we have an array of time steps, an array of density operators at each time step,
and static Qobjs for the Hamiltonians and the Liouvillians.
'''

#The rate of change of Heat calculated as dot(Q) = tr(H*dot(rho))
#The rate of change of Work calculated as dot(W) = tr(dot(H)*rho)
#The rate of change of Energy calculated as dot(U) = dot(W) + dot(Q)

#The rate of change of Entropy calculated as dot(S) = -tr(dot(rho)ln(rho))

#The change of heat calculated a Delta(Q) = Q(t) - Q(0) = int_{0}^{t} dot(Q)dt
#The change of work calculated as Delta(W) = W(t) - W(0) = int_{0}^{t} dot(W)dt

'''
The change of intertanenergy can be calculated in two ways
Delta(U) = U(t)-U(0) = int_{0}^{t} dot(U)dt and 
Delta(U) = tr(H(t)*rho(t)) - tr(H(0)*rho(t))
'''
'''
For the change of entropy this can as 
Delta(S) = S(t) - S(0) = int_{0}^{t} dot(S)dt and 
Delta(S) = -tr(rho(t)ln(rho(t))) - (-tr(rho(t)ln(rho(t))))
'''
def TimeDependence(O1,O2,t): #Using simple conditional logic to encode the stepwise time dependance 
    Tz,Tx =1,1
    T = Tz + Tx # Lenght of one period. Need to clean this implimentation up
    tr =t%T
    if 0<= tr < Tz:
        return O1
    elif Tz<= tr < T:
        return O2
    
def ThrmodynamicRates(Time,Rho,Hz,Hx,LZ,LX,Sx):
    LXSuper = qu.liouvillian(None,LX) #Generate the Liovillian superoperator 
    LZSuper = qu.liouvillian(None,LZ)
    NumberofSteps = len(Time) #number of time steps
    TimeStep = Time[1]-Time[0]
   
    Hlist = [TimeDependence(Hz,Hx,Time[i]) for i in range(NumberofSteps)]
    Llist = [TimeDependence(LZSuper,LXSuper,Time[i]) for i in range(NumberofSteps)]
    
    DotH = [] #Initialise Dot(H) array
    for i in range(NumberofSteps-1):
        DotH.append((TimeDependence(Hz,Hx,Time[i+1]) - TimeDependence(Hz,Hx,Time[i]))/TimeStep)
    DotH.append((Hz-Hx)/TimeStep) #The last element of the derivative has to be calculated separatly 
    #Initialise empty arrays 
    TimeCrystalSignature = []
    DotRho = []
    LogRhoArr = []
    DotQ = []
    DotW = []
    DotU = []
    DotS = []
    
    DimData = Rho[0].dims # data for constructing qobj 
    ShapeData = Rho[0].shape 
    for i in range(NumberofSteps):
        #Calculate the DTC signature and some intermediate values
        TimeCrystalSignature.append((Rho[i]*Sx[0]).tr().real)
        VectorisedState = qu.operator_to_vector(Rho[i])
        DotRho.append(qu.vector_to_operator(Llist[i]*VectorisedState))
        LogRhoArr.append(qu.Qobj(linalg.logm(Rho[i].full()),DimData,ShapeData))
        
        #Calculate the elements of the thermodynamic rates
        DotQ.append((Hlist[i]*DotRho[i]).tr().real)
        DotW.append((DotH[i]*Rho[i]).tr().real)
        DotU.append(DotW[i] + DotQ[i])
        DotS.append(-(DotRho[i]*LogRhoArr[i]).tr().real)
        
    return TimeCrystalSignature,DotQ, DotW, DotU, DotS
'''
These two functions form a simple integration routine. Here I have used the most
basic integral approximation for simplicity as to help deal with the shapnedd of the 
work derivative 
'''
def approxintegral(Arr,Time,dt):
    res = 0
    for i in range(len(Time)):
        res += Arr[i]*dt
    return res

def IntegrationRoutine(Time,Arr,a0): 
    dt = Time[1]-Time[0]
    ApproxInt = []
    for i in range(len(Time)):
        ApproxInt.append(a0 + approxintegral(Arr[:i],Time[:i],dt))
    return ApproxInt

def PlottingWrapper(xArr,yArr,xlbl,ylbl):
    plt.plot(xArr,yArr)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.show()
    return

def AbslistDifference(Arr1,Arr2):
    NewList = []
    for i in range(len(Arr1)):
        NewList.append(abs(Arr1[i] - Arr2[i]))
    return NewList

def IntergratedProperties(Tag,start,stop,inc,params):
    #Tag is the string corresponding to the parameter that needs to be swept
    #I calculate the change over a single period for the first,last and middle period
    
    FirstPeriodHeat = []
    MiddlePeriodHeat = []
    EndPeriodHeat = []
    
    FirstPeriodWork = []
    MiddlePeriodWork = []
    EndPeriodWork = []
    
    FirstPeriodEnergy = []
    MiddlePeriodEnergy = []
    EndPeriodEnergy = []
    
    FirstPeriodEntropy = []
    MiddlePeriodEntropy = []
    EndPeriodEntropy = []
    SweepArr = np.arange(start,stop,inc)
    N = len(SweepArr)
    for i in range(N):
        print('Loop {}/{}'.format(i+1,N))
        params[Tag] = SweepArr[i]
        
        Np = 8 #Number of Periods 
        MidIndex = int(Np/2)
        FinalIndex = Np-1
        TimeStep = 0.01
        Time, Rho = TimeEvolver(Np,TimeStep,params)
        PeriodEnd = int(len(Time)/Np - 1) 
        StepAmount = int(len(Time)/Np) 
        Hz,Hx,LZ,LX,Sx,Sz,Sy = SystemOperators(params)
        TimeCrystalSignature,DotQ, DotW, DotU, DotS = ThrmodynamicRates(Time,Rho,Hz,Hx,LZ,LX,Sx)
    
        DeltaQ = IntegrationRoutine(Time,DotQ,0)
        FirstPeriodHeat.append(DeltaQ[PeriodEnd] - DeltaQ[0])
        MiddlePeriodHeat.append(DeltaQ[MidIndex*StepAmount+PeriodEnd] - DeltaQ[MidIndex*StepAmount])
        EndPeriodHeat.append(DeltaQ[FinalIndex*StepAmount+PeriodEnd] - DeltaQ[FinalIndex*StepAmount])
        
        DeltaW = IntegrationRoutine(Time,DotW,0)
        FirstPeriodWork.append(DeltaW[PeriodEnd] - DeltaW[0])
        MiddlePeriodWork.append(DeltaW[MidIndex*StepAmount+PeriodEnd] - DeltaW[MidIndex*StepAmount])
        EndPeriodWork.append(DeltaW[FinalIndex*StepAmount+PeriodEnd] - DeltaW[FinalIndex*StepAmount])
        
        DeltaU= IntegrationRoutine(Time,DotU,0)
        FirstPeriodEnergy.append(DeltaU[PeriodEnd] - DeltaU[0])
        MiddlePeriodEnergy.append(DeltaU[MidIndex*StepAmount+PeriodEnd] - DeltaU[MidIndex*StepAmount])
        EndPeriodEnergy.append(DeltaU[FinalIndex*StepAmount+PeriodEnd] - DeltaU[FinalIndex*StepAmount])
        
        DeltaS = IntegrationRoutine(Time,DotS,0)  
        FirstPeriodEntropy.append(DeltaS[PeriodEnd] - DeltaS[0])
        MiddlePeriodEntropy.append(DeltaS[MidIndex*StepAmount+PeriodEnd] - DeltaS[MidIndex*StepAmount])
        EndPeriodEntropy.append(DeltaS[FinalIndex*StepAmount+PeriodEnd] - DeltaS[FinalIndex*StepAmount])
        
        HeatChangeArr = [FirstPeriodHeat,MiddlePeriodHeat,EndPeriodHeat]
        WorkChangeArr = [FirstPeriodWork,MiddlePeriodWork,EndPeriodWork]
        EnergyChangeArr = [FirstPeriodEnergy,MiddlePeriodEnergy,EndPeriodEnergy]
        EntropyChangeArr = [FirstPeriodEntropy,MiddlePeriodEntropy,EndPeriodEntropy]
        
    return SweepArr, HeatChangeArr, WorkChangeArr, EnergyChangeArr, EntropyChangeArr

if __name__ == "__main__":
    '''
    The required parameters from the highest level of simulation will be provided in a 
    dictionary and passed along. Closed system parameters listed first, then bath parameters
    For example considering the following
    '''
    params = {"NumSpins":5, "Jmean":1, "Jvar":0.5, "hmean":3,"hvar":0, "ExternalField":0,"Tz":1,"Tx":1, "s":1,"G":0.1,"B":1}
    
    Np = 8 #Number of Periods 
    TimeStep = 0.01
    T = params["Tz"] + params["Tx"]
    Time, Rho = TimeEvolver(Np,TimeStep,params)
    Hz,Hx,LZ,LX,Sx,Sz,Sy = SystemOperators(params)
    TimeCrystalSignature,DotQ, DotW, DotU, DotS = ThrmodynamicRates(Time,Rho,Hz,Hx,LZ,LX,Sx)
    
    PlottingWrapper(Time/T,TimeCrystalSignature,"t/T",r"$\langle \sigma_{1}^{x}(t)\rangle$")
    PlottingWrapper(Time/T,DotQ,"t/T",r"$\dot{Q}(t)$")
    PlottingWrapper(Time/T,DotW,"t/T",r"$\dot{W}(t)$")
    PlottingWrapper(Time/T,DotU,"t/T",r"$\dot{U}(t)$")
    PlottingWrapper(Time/T,DotS,"t/T",r"$\dot{S}(t)$")
    
    '''
    To verify that the previous integration routine makes sense to some degree
    I can calculate the change of internal energy and entropy via a different route 
    and compare the results
    
    Delta(U) = tr(H*rho)
    Delta(S) = -tr(rho*ln(rho))
    '''
    EntropyTestArr = []
    EnergyTestArr = []
    for i in range(len(Rho)):
        EntropyTestArr.append(qu.entropy_vn(Rho[i])-qu.entropy_vn(Rho[0]))
        EnergyTestArr.append((TimeDependence(Hz,Hx,Time[i])*Rho[i]).tr() - (TimeDependence(Hz,Hx,Time[0])*Rho[0]).tr())
    
    DeltaQ = IntegrationRoutine(Time,DotQ,0)
    DeltaW = IntegrationRoutine(Time,DotW,0)
    DeltaU= IntegrationRoutine(Time,DotU,0)
    DeltaS = IntegrationRoutine(Time,DotS,0)
    
    PlottingWrapper(Time/T,DeltaQ,"t/T",r"$\Delta Q(t)$")
    PlottingWrapper(Time/T,DeltaW,"t/T",r"$\Delta W(t)$")
    PlottingWrapper(Time/T,DeltaU,"t/T",r"$\Delta U(t)$")
    PlottingWrapper(Time/T,DeltaS,"t/T",r"$\Delta S(t)$")
    
    PlottingWrapper(Time/T,AbslistDifference(DeltaS,EntropyTestArr),"t/T","Error in S")
    PlottingWrapper(Time/T,AbslistDifference(DeltaU,EnergyTestArr),"t/T","Error in U")
    
    '''
    Not I want to be able to look at the integrated quantities over different periods of the evolution
    This routine should take a given set of system parameters and a period of interest
    then return the intergrated values
    '''
    
    SweepArr, HeatChangeArr, WorkChangeArr, EnergyChangeArr, EntropyChangeArr = IntergratedProperties("G",0.000001,2.01,0.01,params)
    
    PlottingWrapper(SweepArr,HeatChangeArr[0],r"$\Gamma$",r"$\Delta Q$")
    PlottingWrapper(SweepArr,WorkChangeArr[0],r"$\Gamma$",r"$\Delta W$")
    PlottingWrapper(SweepArr,EnergyChangeArr[0],r"$\Gamma$",r"$\Delta U$")
    PlottingWrapper(SweepArr,EntropyChangeArr[0],r"$\Gamma$",r"$\Delta S$")
    
    PlottingWrapper(SweepArr,HeatChangeArr[1],r"$\Gamma$",r"$\Delta Q$")
    PlottingWrapper(SweepArr,WorkChangeArr[1],r"$\Gamma$",r"$\Delta W$")
    PlottingWrapper(SweepArr,EnergyChangeArr[1],r"$\Gamma$",r"$\Delta U$")
    PlottingWrapper(SweepArr,EntropyChangeArr[1],r"$\Gamma$",r"$\Delta S$")
    
    PlottingWrapper(SweepArr,HeatChangeArr[2],r"$\Gamma$",r"$\Delta Q$")
    PlottingWrapper(SweepArr,WorkChangeArr[2],r"$\Gamma$",r"$\Delta W$")
    PlottingWrapper(SweepArr,EnergyChangeArr[2],r"$\Gamma$",r"$\Delta U$")
    PlottingWrapper(SweepArr,EntropyChangeArr[2],r"$\Gamma$",r"$\Delta S$")
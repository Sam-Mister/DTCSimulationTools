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


from TimeEvolver import TimeEvolver
from TimeEvolver import SystemOperators
from scipy import linalg
import qutip as qu
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import integrate
Standardparams = {"seed":42,"NumSpins":5, "Jmean":np.pi/4, "Jvar":0, "hmean":np.pi,"hvar":0, "ExternalField":0,"Tz":1,"Tx":1, "s":1,"G":0.001,"B":1}
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

def IntegrationRoutine(Time,Arr,a0,params): 
    #This routine calculates the integrated properties at integer multiples of T
    T = params["Tz"] + params["Tx"]
    dt = Time[1]-Time[0]
    StepSamples = int(T/dt)
    Ptime = np.arange(0,Time[-1]+T,T)
    ApproxInt = [0]
    for i in range(1,len(Ptime)):
        ApproxInt.append(a0 + integrate.trapezoid(Arr[:(i*StepSamples-1)], Time[:(i*StepSamples-1)]))
        #ApproxInt.append(a0 + approxintegral(Arr[:i],Time[:i],dt))
    return Ptime, ApproxInt

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
        t0 = time.time()
        print('Loop {}/{}'.format(i+1,N))
        params[Tag] = SweepArr[i]
        
        Np = 8 #Number of Periods 
        MidIndex = int(Np/2)
        FinalIndex = Np-1
        TimeStep = 0.0005
       
        Hz,Hx,LZ,LX,Sx,Sz,Sy = SystemOperators(params)
        print('Generating Initial State')
        psi0 = qu.qload('limitstate')#InitialStateGenerator(Standardparams)#qu.tensor([qu.Qobj([[1/np.sqrt(2)],[1/np.sqrt(2)]]) for n in range(params["NumSpins"])])
        print('Initial State Generated')
        Time, Rho = TimeEvolver(Np,TimeStep,psi0,params)
        PeriodEnd = int(len(Time)/Np - 1) 
        StepAmount = int(len(Time)/Np)
        TimeCrystalSignature,DotQ, DotW, DotU, DotS = ThrmodynamicRates(Time,Rho,Hz,Hx,LZ,LX,Sx)
    
        PTime,DeltaQ = IntegrationRoutine(Time,DotQ,0,params)
        FirstPeriodHeat.append(DeltaQ[1] - DeltaQ[0])
        MiddlePeriodHeat.append(DeltaQ[MidIndex+1] - DeltaQ[MidIndex])
        EndPeriodHeat.append(DeltaQ[FinalIndex+1] - DeltaQ[FinalIndex])
        
        PTime,DeltaW = IntegrationRoutine(Time,DotW,0,params)
        FirstPeriodWork.append(DeltaW[1] - DeltaW[0])
        MiddlePeriodWork.append(DeltaW[MidIndex+1] - DeltaW[MidIndex])
        EndPeriodWork.append(DeltaW[FinalIndex+1] - DeltaW[FinalIndex])
        
        PTime,DeltaU= IntegrationRoutine(Time,DotU,0,params)
        FirstPeriodEnergy.append(DeltaU[1] - DeltaU[0])
        MiddlePeriodEnergy.append(DeltaU[MidIndex+1] - DeltaU[MidIndex])
        EndPeriodEnergy.append(DeltaU[FinalIndex+1] - DeltaU[FinalIndex])
        
        PTime,DeltaS = IntegrationRoutine(Time,DotS,0,params)  
        FirstPeriodEntropy.append(DeltaS[1] - DeltaS[0])
        MiddlePeriodEntropy.append(DeltaS[MidIndex+1] - DeltaS[MidIndex])
        EndPeriodEntropy.append(DeltaS[FinalIndex+1] - DeltaS[FinalIndex])
        t1 = time.time()
        print('This loop took {:.2}mins'.format((t1-t0)/60))
        print('Approx {:.2} mins left'.format(((N-(i+1))*(t1-t0))/60))
    HeatChangeArr = [FirstPeriodHeat,MiddlePeriodHeat,EndPeriodHeat]
    WorkChangeArr = [FirstPeriodWork,MiddlePeriodWork,EndPeriodWork]
    EnergyChangeArr = [FirstPeriodEnergy,MiddlePeriodEnergy,EndPeriodEnergy]
    EntropyChangeArr = [FirstPeriodEntropy,MiddlePeriodEntropy,EndPeriodEntropy]
        
    return SweepArr, HeatChangeArr, WorkChangeArr, EnergyChangeArr, EntropyChangeArr

def WorkCal(Hx,Hz,Rho,Time,Tx,Tz,t):
    T =Tx + Tz
    n = int(np.floor(t/T))
    r = t%T
    workSum = 0 
    TG = int(len(Time)/Np)
    for i in range(1,n+1):
        index1 = int(i*TG - 1)
        index2 = int(TG*(i-(1/2)) - 1)
        workSum += ((Hz-Hx)*(Rho[index1] - Rho[index2])).tr()
    if 0 <= r < Tz: 
        return workSum
    elif Tz<=r<T:
        finalIndex = int(TG*(n+(1/2))-1)
        return workSum - ((Hz-Hx)*Rho[finalIndex]).tr()
    
def AlternativeIntrgation(params):
    #This is an alternative way to calculate the integrated quantities that should be faster and more accurate
    Hz,Hx,LZ,LX,Sx,Sz,Sy = SystemOperators(params)
    print('Generating Initial State')
    psi0 = qu.qload('limitstate')#InitialStateGenerator(Standardparams)#qu.tensor([qu.Qobj([[1/np.sqrt(2)],[1/np.sqrt(2)]]) for n in range(params["NumSpins"])])
    print('Initial State Generated')
    Time, Rho = TimeEvolver(Np,TimeStep,psi0,params)
    Entropy = []
    Energy = []
    Work = []
    Heat = []
    for i in range(len(Rho)):
        Entropy.append(qu.entropy_vn(Rho[i]))
        Energy.append((TimeDependence(Hz,Hx,Time[i])*Rho[i]).tr().real)
        Work.append(WorkCal(Hx,Hz,Rho,Time,params["Tx"],params["Tz"],Time[i]).real)
        Heat.append(Energy[i] - Work[i])
    return Time, Entropy, Energy, Work, Heat

def IntergratedProperties2(Tag,start,stop,inc,params):
    
    EndPeriodHeat = []
    
    EndPeriodWork = []
    
    EndPeriodEnergy = []

    EndPeriodEntropy = []
    SweepArr = np.arange(start,stop,inc)
    N = len(SweepArr)
    for i in range(N):
        print('Loop {}/{}'.format(i+1,N))
        params[Tag] = SweepArr[i]
        Time,Entropy, Energy, Work, Heat = AlternativeIntrgation(params)
        TG = int(len(Time)/Np)
        
        EndPeriodHeat.append(Heat[int(Np*TG)-1] - Heat[int((Np-1)*TG)-1])
    
        EndPeriodWork.append(Work[int(Np*TG)-1] - Work[int((Np-1)*TG)-1])
    
        EndPeriodEnergy.append(Energy[int(Np*TG)-1] - Energy[int((Np-1)*TG)-1])
        
        EndPeriodEntropy.append( Entropy[int((Np-1)*TG)+1]- Entropy[int(Np*TG)-1] )

    return SweepArr,EndPeriodHeat,EndPeriodWork,EndPeriodEnergy,EndPeriodEntropy
def InitialStateGenerator(params):
    psi0 = qu.tensor([qu.Qobj([[1/np.sqrt(2)],[1/np.sqrt(2)]]) for n in range(params["NumSpins"])])
    Time, Rho = TimeEvolver(8,0.1,psi0,params)
    Init = Rho[-1]
    return Init

def CoherenceRemover(Qobj):
    dimentions = Qobj.dims
    shape = Qobj.shape
    data = Qobj.full()
    N = data.shape[0]
    for i in range(N):
        for j in range(N):
            if i !=j:
                data[i,j] = 0 
    NewObj = qu.Qobj(data,dimentions,shape)
    return NewObj

def ZEvolutionCals(r0,params):
    params["G"] = 20
    Hz,Hx,LZ,LX,Sx,Sz,Sy = SystemOperators(params)
    tlist = np.arange(0,1,0.05)
    results = qu.mesolve(Hz, r0, tlist,LZ,[])
    return results.states[-1
                          ]
if __name__ == "__main__":
    '''
    The required parameters from the highest level of simulation will be provided in a 
    dictionary and passed along. Closed system parameters listed first, then bath parameters
    For example considering the following
    '''
    params = {"seed":42,"NumSpins":5, "Jmean":np.pi/4, "Jvar":0.5, "hmean":np.pi,"hvar":0.1, "ExternalField":0,"Tz":1,"Tx":1, "s":1,"G":10,"B":1}
    
    Np = 8 #Number of Periods 
    TimeStep = 0.01
    T = params["Tz"] + params["Tx"]
    
    print('Generating Initial State')
    psi0 = InitialStateGenerator(Standardparams)#qu.tensor([qu.Qobj([[1/np.sqrt(2)],[1/np.sqrt(2)]]) for n in range(params["NumSpins"])])
    qu.qsave(psi0, 'limitstate')
    print('Initial State Generated')
    print('--------------')
    Hz,Hx,LZ,LX,Sx,Sz,Sy = SystemOperators(params)
    
    print('Evolving the system')
    Time, Rho = TimeEvolver(Np,TimeStep,psi0,params)
    print('System Evolved')
    print('--------------')
    psi0Therm = (-params["B"]*Hx).expm()/((-params["B"]*Hx).expm()).tr()
    rho_infty = psi0Therm
    rho_inftyZ = ZEvolutionCals(rho_infty,params)
    Fidthermal = []
    FidDiagonal = []
    for i in range(len(Rho)):
        Fidthermal.append(qu.fidelity(Rho[i].tidyup(),rho_infty))
        FidDiagonal.append(qu.fidelity(Rho[i].tidyup(),rho_inftyZ))
    plt.plot(Time,Fidthermal,label = 'Thermal')
    plt.plot(Time,FidDiagonal)
    plt.legend()
    plt.show()
    
    
    print('Calculating Thrmodynamic Rates')
    TimeCrystalSignature,DotQ, DotW, DotU, DotS = ThrmodynamicRates(Time,Rho,Hz,Hx,LZ,LX,Sx)
    print('Done')
    print('--------------')
    
    PlottingWrapper(Time/T,TimeCrystalSignature,"t/T",r"$\langle \sigma_{1}^{x}(t)\rangle$")
    PlottingWrapper(Time/T,DotQ,"t/T",r"$\dot{Q}(t)$")
    PlottingWrapper(Time/T,DotW,"t/T",r"$\dot{W}(t)$")
    PlottingWrapper(Time/T,DotU,"t/T",r"$\dot{U}(t)$")
    PlottingWrapper(Time/T,DotS,"t/T",r"$\dot{S}(t)$")
    
    
    print('Calculating Approximate Integrated Quantities')
    PTime, DeltaQ = IntegrationRoutine(Time,DotQ,0,params)
    PTime,DeltaW = IntegrationRoutine(Time,DotW,0,params)
    PTime,DeltaU= IntegrationRoutine(Time,DotU,0,params)
    PTime, DeltaS = IntegrationRoutine(Time,DotS,0,params)
    print('Done')
    print('--------------')
    
    print("Calculating Integrated quants via direct methods")
    Time,Entropy, Energy, Work, Heat = AlternativeIntrgation(params)
    plt.plot(Time/T,Entropy)
    plt.plot(PTime/T,DeltaS)
    plt.xlabel(r"t/T")
    plt.ylabel(r"$\Delta S$")
    plt.show()
    plt.plot(Time/T,Energy)
    plt.plot(PTime/T,DeltaU)
    plt.xlabel(r"t/T")
    plt.ylabel(r"$\Delta U$")
    plt.show()
    plt.plot(Time/T,Work)
    plt.plot(PTime/T,DeltaW)
    plt.xlabel(r"t/T")
    plt.ylabel(r"$\Delta W$")
    plt.show()
    plt.plot(Time/T,Heat)
    plt.plot(PTime/T,DeltaQ)
    plt.xlabel(r"t/T")
    plt.ylabel(r"$\Delta Q$")
    plt.show()
    
    #print(Work[-1]-Work[])
    
    '''
    Now I will calculate the analytic expressions for the large gamma limit. To do this I need the thermal state of Hx, this state with not coherence in the 
    z basis and the hamiltonions Hx,Hz
    '''
    
    
    WorkLimit = ((Hz-Hx)*(rho_infty-rho_inftyZ)).tr().real
    HeatLimit = -WorkLimit
    EnergyLimit = 0
    EntropyLimit = qu.entropy_vn(rho_inftyZ) - qu.entropy_vn(rho_infty) #qu.entropy_relative(rho_infty,rho_inftyZ)+ qu.entropy_relative(rho_inftyZ,rho_infty)
    print(WorkLimit)
    print(HeatLimit)
    print(EnergyLimit)
    print(EntropyLimit)
    
    
    '''
    Not I want to be able to look at the integrated quantities over different periods of the evolution
    This routine should take a given set of system parameters and a period of interest
    then return the intergrated values
    '''
    
    print('Begin looping over different enviroment coupling strengths')
    SweepArr, HeatChangeArr, WorkChangeArr, EnergyChangeArr, EntropyChangeArr = IntergratedProperties2("G",0.00000001,10.01,0.5,params)
    
    AsyHeat = []
    AsyWork = []
    AsyEng = []
    AsyEnt = []
    for i in range(len(SweepArr)):
        AsyHeat.append(HeatLimit)
        AsyWork.append(WorkLimit)
        AsyEng.append(EnergyLimit)
        AsyEnt.append(EntropyLimit)
        
    plt.plot(SweepArr,HeatChangeArr,label = r"$\Delta Q$")
    plt.plot(SweepArr,WorkChangeArr,label = r"$\Delta W$")
    plt.plot(SweepArr,EnergyChangeArr,label = r"$\Delta U$")
    plt.plot(SweepArr,EntropyChangeArr,label = r"$\Delta S$")
    plt.plot(SweepArr,AsyHeat,"-")
    plt.plot(SweepArr,AsyWork,"-")
    plt.plot(SweepArr,AsyEng,"-")
    plt.plot(SweepArr,AsyEnt,"-")
    plt.legend()
    plt.show()
    
   
   
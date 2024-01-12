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

from TimeEvolver import TimeEvolver
from TimeEvolver import SystemOperators
from scipy import linalg
import qutip as qu

Np = 5 #Number of Periods 
TimeStep = 0.1

Time, Rho = TimeEvolver(Np,TimeStep)
Hz,Hx,LZ,LX,Sx, Sz, Sy = SystemOperators()
NumberofSteps = len(Time) #number of time steps

LXSuper = qu.liouvillian(None,LX) #Generate the Liovillian superoperator 
LZSuper = qu.liouvillian(None,LZ)
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
def TimeDependence(O1,O2,t):
    Tz,Tx =1,1
    T = Tz + Tx # Lenght of one period. Need to clean this implimentation up
    tr =t%T
    if 0<= tr < Tz:
        return O1
    elif Tz<= tr < T:
        return O2
    
Hlist = [TimeDependence(Hz,Hx,Time[i]) for i in range(NumberofSteps)]
Llist = [TimeDependence(LZSuper,LXSuper,Time[i]) for i in range(NumberofSteps)]
DotH = [(Hz-Hx)/TimeStep]
for i in range(NumberofSteps-1):
    DotH.append((TimeDependence(Hz,Hx,Time[i+1]) - TimeDependence(Hz,Hx,Time[i]))/TimeStep)

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
    TimeCrystalSignature.append((Rho[i]*Sx[0]).tr().real)
    VectorisedState = qu.operator_to_vector(Rho[i])
    DotRho.append(qu.vector_to_operator(Llist[i]*VectorisedState))
    LogRhoArr.append(qu.Qobj(linalg.logm(Rho[i].full()),DimData,ShapeData))
    DotQ.append((Hlist[i]*DotRho[i]).tr().real)
    DotW.append((DotH[i]*Rho[i]).tr().real)
    DotU.append(DotW[i] + DotQ[i])
    DotS.append(-(DotRho[i]*LogRhoArr[i]).tr().real)




    
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

Np = 5 #Number of Periods 
TimeStep = 0.1

Time, Rho = TimeEvolver(Np,TimeStep)
Hz,Hx,LZ,LX = SystemOperators()

'''
Now we have an array of time steps, an array of density operators at each time step,
and static Qobjs for the Hamiltonians and the Liouvillians.
'''


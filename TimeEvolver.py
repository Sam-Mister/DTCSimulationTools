# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:32:12 2024

@author: sm15883
"""
import numpy as np
from SystemToolbox import PauliN
from SystemToolbox import disorderGen
from SystemToolbox import HamiltonianGenerator
from SystemToolbox import GeneralGlobalME 

NumberOfSpins = 5
Jmean = 1
Jvar = 0
hmean = np.pi
hvar = 0.5

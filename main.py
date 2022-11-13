#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 13:39:20 2022

@author: simonl
"""

#%% dependencies
import numpy as np
from copy import deepcopy
import aa
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import gamma
from scipy import optimize
import time
import sys
import os
import matplotlib.colors as mcolors
import matplotlib as mpl
import seaborn as sns
from classes import moments, parameters, cobweb, var, sol_class, history
# sns.set()
# sns.set_context('talk')
# sns.set_style('white')

#%% build dic compare moments





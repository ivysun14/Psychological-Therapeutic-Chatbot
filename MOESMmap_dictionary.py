#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 21:49:26 2023

@author: maguo
"""

import json
import pandas as pd

MOES = pd.read_csv("13428_2013_403_MOESM1_ESM.csv")

data = json.load(open('processed/meta.json'))

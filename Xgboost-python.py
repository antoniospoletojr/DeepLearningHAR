#!/usr/bin/env python
# coding: utf-8

# # HAR CNN training 

# In[1]:


# Imports
import numpy as np
import os
import tensorflow as tf
from utils.utilities import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from IPython import get_ipython
import xgboost as xgb

feat_names = pd.read_csv("./UCIHAR/features.txt", sep=" ", names=["code","feature"])
#feat_names["code"] = "f"+feat_names["code"]
#feat_names

if __name__ == "__main__":
   print("ciao")

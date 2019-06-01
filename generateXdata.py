
# IMport dependencies
import pandas as pd
from pandas import set_option
import numpy as np
import os
import csv

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# from keras.utils import to_categorical

# import preprocessing from sklearn
from sklearn import preprocessing

# DictVectorizer
from sklearn.feature_extraction import DictVectorizer


#############################################################################################################
# Fuction to generate X predictor array or dataframe from raw data and key features columns required
# Accepts Two parameters
# ohe : One Hot Encoding required
# target : BTU or DOLLAR. Based on the target being predicted the target columns will  be dropped / retained
#############################################################################################################

def generateX(ohe = True, target = "BTU"):    

    dataFilePath = "dataforfinalproject"
    filename = "RECS_COMBINED_DATA.csv"
    cols_file = "Final_Columns_withCat.csv"

    # read dataset wih all years combined data
    df_recs = pd.read_csv(os.path.join(dataFilePath, filename), low_memory= False)


    # read the columns from Columns csv
    df_cols = pd.read_csv(os.path.join(dataFilePath, cols_file))
#     df_cols.columns

    # Whittle down the dataset to contain only Features required for modeling - X 
    modelDF = df_recs[df_cols[df_cols.FEATURES_MODEL == "Y"].COLUMN_NAME]
    print(f" X Features shape : {modelDF.shape}")

    y_label = df_recs['TOTALBTU']
    print(f"y label shape : {y_label.shape}")

    
    if(target == "BTU"):
        # Drop Price / Cost related Columns as it is only Consumption we are interested in 
        cost_cols = df_cols[(df_cols['COLUMN_NAME'].str.find("DOL") != -1) & (df_cols.FEATURES_MODEL == "Y")].COLUMN_NAME.tolist()
        modelDF.drop(cost_cols, axis = 1, inplace = True)
        # Drop All BTU related cols too
        btu_cols = df_cols[(df_cols['COLUMN_NAME'].str.find("BTU") != -1) & (df_cols.FEATURES_MODEL == "Y")].COLUMN_NAME.tolist()
        modelDF.drop(btu_cols, axis = 1, inplace = True)
        
        # and drop TOTAL BTU from X set
#         X = modelDF.drop(['TOTALBTU'], axis = 1)
    else:
        # Drop Price / Cost related Columns as it is only Consumption we are interested in 
        cost_cols = df_cols[(df_cols['COLUMN_NAME'].str.find("DOL") != -1) & (df_cols.FEATURES_MODEL == "Y")].COLUMN_NAME.tolist()
        modelDF.drop(cost_cols, axis = 1, inplace = True)
        
        # Also drop the Total BTU cols 
        btu_cols = df_cols[(df_cols['COLUMN_NAME'].str.find("TOTALBTU") != -1) & (df_cols.FEATURES_MODEL == "Y")].COLUMN_NAME.tolist()
        modelDF.drop(btu_cols, axis = 1, inplace = True)


            # and drop TOTAL BTU from X set
#         X = modelDF.drop(['TOTALDOLLAR'], axis = 1)
        
    X = modelDF
    print(f"shape of X is {X.shape}")

    if(ohe):
        ### Apply dict vectorizer 
        # convert the X array into a dict
        X_dict = X.to_dict(orient = "records")
       

        # instantiate a Dictvectorizer object for X
        dv_X = DictVectorizer(sparse=False)   # sparse = False makes the output is not a sparse matrix

        # apply dv_X on X_dict
        X_encoded = dv_X.fit_transform(X_dict)
        
        vocab = dv_X.get_feature_names()
        # return X_encoded and its vocab
        return (X_encoded, vocab)
    else:
        return (X, X.columns)


# In[4]:


# generateX()


# In[ ]:





# PY file that contains function to generate price for given region and sqft range

# Generic Dependencies
import numpy as np
from numpy import arange
import pandas as pd
from pandas import set_option
# from pandas.tools.plotting import scatter_matrix
import os

# Plotting Libraries
from matplotlib import pyplot as plt
import seaborn as sns

# SKLearn Libraries
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso, LassoCV, Ridge,RidgeCV
from sklearn.model_selection import cross_validate

# library for saving models
from sklearn.externals import joblib

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

# DictVectorizer
from sklearn.feature_extraction import DictVectorizer

#XGBoost libraries
import xgboost as xgb

# explain xgboost preiction and model
import shap


# import generateXdata.py to prepare the data
# from generateXdata import generateX
from generateXdata import generateX_samp

#Import Custom ml models
from Cls_ml_models import ml_models_tuned


def predictPrice(selRegion, selSQFT):
    
    # gobal variables
    dataFilePath = "dataforfinalproject"    

    X, vocab, y = generateX_samp(ohe = True, target = "DOLLAR", numSamples = 30, region = selRegion, totsqft_cd = selSQFT)
    
    print(X.shape)
    print(y.shape)
    
    # load the model into a list
    model_name = ["Classic Lasso","Elasticnet", "LassoCV", "LinearRegression","RandomForest", "RidgeCV","XGBoost"]
    models = [joblib.load(os.path.join("final_models",file)) for file in os.listdir("final_models") if file.endswith("sav")]

    results_data = pd.DataFrame(columns = ["Actual"])
    results_score = pd.DataFrame(columns = ["Model","R2", "RMSE"])

    results_data['Actual'] = y
#     results_data

    for i, model in enumerate(models):
        print(model_name[i])
    #     print(X.iloc[0,:])
        pred_price = model.predict(X)
        r2 = round(model.score(X, y)*100,2)
        rmse = round(np.sqrt(mean_squared_error(y,pred_price)))

        print(f"Predicted Price is {pred_price}")
        print(f"R2 value is {r2}")

        results_data[model_name[i]] = pred_price

        results_score['Model'] = model_name[i]
        results_score['R2'] = r2
        results_score['RMSE'] = rmse




    # save results into a CSV for further plotting / review
    results_data.reset_index().to_csv(os.path.join(dataFilePath,"resultsdata.csv"), index = False)
    
    print(results_data)

    # reshape results data for comparison plotting
    results1 = pd.DataFrame(results_data.stack(), columns = ["Price"])

    results1.reset_index(inplace = True)

    results1.columns = ['SampleNum','Model','Price']

    # save the reshaped file to csv for plots
    results1.to_csv(os.path.join(dataFilePath,"resultsdata_1.csv"))
    
    return results_data[['Classic Lasso','Elasticnet','LassoCV','LinearRegression','RandomForest','RidgeCV','XGBoost']].to_html(table_id = "results", classes = "table table-striped")

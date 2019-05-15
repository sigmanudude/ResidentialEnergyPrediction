
# Generic Dependencies
import numpy as np
from numpy import arange
import pandas as pd
from pandas import set_option
from pandas.tools.plotting import scatter_matrix
import os


# SKLearn - helper libraries
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_validate, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error


# SKLearn - Linear Model libraries
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, LassoCV

# SKLearn - Tree Boosting libraries
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb



class ml_models_tuned(object):
    def __init__(self, X, y, seed, folds, test_size, early_stop):
        self.X = X
        self.y = y
        self.seed = seed
        self.folds = folds
        self.test_size = test_size
        self.early_stop = early_stop
        
        self.dMatrix = xgb.DMatrix(data = self.X,label = self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state = self.seed, 
                                                                              test_size = self.test_size)
    def ret_LR_tunedModel(self):        
        model = LinearRegression()
        print(f"Best Linear Regression model {model}")
        return model
    
    def ret_Classo_tunedModel(self):
        print("\n Begining to tune Lasso Regression Model.................................\n")
        # perform a grid search with different alpha values to find the best fit
#         model = Lasso()
#         alpha=np.arange(0.0,0.05,.003)
        
#         param_grid = [{'alpha':alpha}]
#         pricing_grid  = GridSearchCV(model, param_grid, cv=5,scoring='r2')
#         pricing_grid.fit(self.X_train, self.y_train)
        
#         model = pricing_grid.best_estimator_
        model = Lasso(alpha=0.01)
        print(f"Best Lasso model with alpha = 0.01 is {model}")
        return model
    
    def ret_LassoCV_tunedModel(self):
#         lasso = Lasso(max_iter=10000, normalize=True)
        lassocv = LassoCV(alphas=None, cv=10, max_iter=100000, normalize=True)
        lassocv.fit(self.X_train, self.y_train)        
#         lasso.set_params(alpha=lassocv.alpha_)
        
        model = lassocv
        print(f"Best LassoCV model with alpha {lassocv.alpha_} is {model}")
        return model
    
    def ret_RF_tunedModel(self):
        model = RandomForestRegressor(n_estimators = 50, random_state = self.seed, max_depth = 11)
        print(f"Best RF model obtained is with n_estimators at 50 and max_depth at 11 is {model}")
        return model
    
    def ret_xgb_tunedModel(self):
#         model = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.8, learning_rate = 0.25,min_child_weight=3,
#                 max_depth = 5, alpha = 10, n_estimators = 150, subsample = 0.8, silent = 1)
        
#         xgb_param = model.get_xgb_params()
#         cvresult = xgb.cv(params = xgb_param, dtrain = self.dMatrix, num_boost_round=model.get_params()['n_estimators'], 
#                           nfold=self.folds, metrics='rmse', early_stopping_rounds=self.early_stop, verbose_eval=False)
        
#         model.set_params(n_estimators=cvresult.shape[0])
        model = xgb.XGBRegressor(alpha=10, base_score=0.5, booster='gbtree',
       colsample_bylevel=1, colsample_bytree=0.9, gamma=0,
       importance_type='gain', learning_rate=0.05, max_delta_step=0,
       max_depth=8, min_child_weight=5, missing=None, n_estimators=300,
       n_jobs=1, nthread=2, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=42, silent=1,
       subsample=0.7)
        
        print(f"Best xgBoost model obtained is with n_estimators at {model.get_params()['n_estimators']} is {model}")
    
        return model
    
    def ret_eNet_tunedModel(self):
        enet = ElasticNet(alpha=0.003, copy_X=True, fit_intercept=True, l1_ratio=0.5,
         max_iter=1000, normalize=False, positive=False, precompute=False,
         random_state=None, selection='cyclic', tol=0.0001, warm_start=False)
        
        print(f"Best elasticNet model obtained is {enet}")
        return enet
        
    def linearmodel(self, model):
        print("Begining to run Linear regression Model.................................\n")
                
        print(f"Model used for Fitting and Predicting is {model}.....................\n")
#         model = self.ret_LR_tunedModel()
        
        
        print("Fit train data to the model.....................\n")
        model.fit(self.X_train, self.y_train)
        
        print("Predict Y values for test data.....................\n")
        predictions_train = model.predict(self.X_train)
        predictions= model.predict(self.X_test)
        
        RMSE_train = np.sqrt(mean_squared_error(self.y_test, predictions))
        r2_train = model.score(self.X_test, self.y_test)
        
        RMSE_test = np.sqrt(mean_squared_error(self.y_test, predictions))
        r2_test = model.score(self.X_test, self.y_test)
        
        column_series = list(predictions)
        df_modelresults=pd.DataFrame()
        df_modelresults = df_modelresults.assign(LinearPredictedY=column_series)
        
        print("Linear Regression Model report")
        print("----------------------------------------------------------------\n")
        
        print("Test RMSE: %f" % (RMSE_test))
        print('The accuracy of the linear regressor is {:.2f} out of 1 on the testing data'.format(r2_test))
        
#         plt.scatter(predictions_train, predictions_train - self.y_train, c="blue", label="Training Data")
#         plt.scatter(predictions, predictions - self.y_test, c="orange", label="Testing Data")
#         plt.legend()
#         plt.hlines(y=0, xmin=self.y_test.min(), xmax=self.y_test.max())
#         plt.title("Residual Plot")
#         plt.show()
        
                
        return(predictions, RMSE_test, r2_test)
    
    def classic_lasso_model(self, model):
        print("\nBegining to run Classic Lasso regression Model (alpha = 0.01).................................\n")                
        
        print(f"Model used for Fitting and Predicting is {model}.....................\n")
        lasso = model.fit(self.X_train, self.y_train)
        
        print("Predict y values for train and test data.....................\n")
        predictions_train = lasso.predict(self.X_train)
        
        RMSE_train = np.sqrt(mean_squared_error(self.y_train, predictions_train))
        r2_train = lasso.score(self.X_train, self.y_train)
        
        predictions = lasso.predict(self.X_test)
        RMSE_test = np.sqrt(mean_squared_error(self.y_test, predictions))
        r2_test= lasso.score(self.X_test, self.y_test)
        
        print("Classic Lasso Regresion Model report")
        print("----------------------------------------------------------------\n")
        print("Train RMSE: %f" % (RMSE_train))
        print("Test RMSE: %f" % (RMSE_test))
        print('The accuracy of the Classic Lasso regressor is {:.2f} out of 1 on the training data'.format(r2_train))
        print('The accuracy of the Classic Lasso regressor is {:.2f} out of 1 on the test data'.format(r2_test))
        
        df_modelresults=pd.DataFrame(predictions,columns=['Classic_Lasso_predicted'])

        return(df_modelresults, RMSE_test,r2_test)
    
    def lasso_CV(self, model):
        print("\nBegining to run Lasso CV regression Model (alpha = determined by K-Folds).................................\n")                
        print(f"Model used for Fitting and Predicting is {model}.....................\n")
        
#         X_train, X_test , y_train, y_test = cross_validation.train_test_split(X, y, test_size=test_size, random_state=seed)
        lasso = model
        
        print("Fit the best model from k-folds to training data................")
        lasso.fit(self.X_train, self.y_train)
        print("best model coefficients:")
        print(pd.Series(lasso.coef_).tolist())
        
        print("\nPredict data using testing data................")
        predictions = lasso.predict(self.X_test)
        RMSE_test = np.sqrt(mean_squared_error(self.y_test, predictions))
        r2 = lasso.score(self.X_test, self.y_test)
        
        print("Lasso CV Regresion Model report")
        print("----------------------------------------------------------------\n")
        
        print("Test RMSE: %f" % (RMSE_test))
        print('The accuracy of the LassoCV regressor is {:.2f} out of 1 on the test data'.format(r2))
        
        df_modelresults=pd.DataFrame(predictions,columns=['Lasso_CV_predicted'])
        
        return(df_modelresults,RMSE_test,r2)

    def RF_model(self, model):
        print("\nBegining to run RandomForest regression Model.................................\n")                
        print(f"Model used for Fitting and Predicting is {model}.....................\n")

        rf = model

        # Train the model on training data
        rf.fit(self.X_train, self.y_train)

        print("Predict y values for train and test data.....................\n")
        predictions = rf.predict(self.X_test)
        RMSE_test = np.sqrt(mean_squared_error(self.y_test, predictions))
        r2_test = rf.score(self.X_test, self.y_test)

        predictions_train = rf.predict(self.X_train)
        RMSE_train = np.sqrt(mean_squared_error(self.y_train, predictions_train))
        r2_train = rf.score(self.X_train, self.y_train)

        print("RandomForest Regresion Model report")
        print("----------------------------------------------------------------\n")
        print("Train RMSE: %f" % (RMSE_train))
        print("Test RMSE: %f" % (RMSE_test))
        print('The accuracy of the RandomForest regressor is {:.2f} out of 1 on the training data'.format(r2_train))
        print('The accuracy of the RandomForest regressor is {:.2f} out of 1 on the test data'.format(r2_test))

        df_modelresults=pd.DataFrame(predictions,columns=['RandomForest_predictions'])

        return(df_modelresults, RMSE_test,r2_test)

    def xgb_model(self, model):
        print("Begining to run XGBoost Model.................................\n")
                
        print(f"Model used for Fitting and Predicting is {model}.....................\n")
        
#         model = self.ret_xgb_tunedModel()
        print("Fitting Train data to the model ")
        print("----------------------------------------------------------------\n")
        eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
        model.fit(self.X_train, self.y_train,eval_metric=["rmse"], eval_set=eval_set, verbose=False)
        
        print("Predicting Y with test data")
        print("----------------------------------------------------------------\n")
        preds_train = model.predict(self.X_train)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, preds_train))
        train_accur = model.score(self.X_train, self.y_train)

        preds = model.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, preds))
        accur = model.score(self.X_test, self.y_test)
        
        
        print("XGBoost Model report")
        print("----------------------------------------------------------------\n")
        print("Train RMSE: %f" % (train_rmse))
        print("Test RMSE: %f" % (rmse))
        print('The accuracy of the xgboost regressor is {:.2f} out of 1 on the training data'.format(train_accur))
        print('The accuracy of the xgboost regressor is {:.2f} out of 1 on the test data'.format(accur))
    
#         # retrieve performance metrics and plot it
#         results = model.evals_result()
#         epochs = len(results['validation_0']['rmse'])
#         x_axis = range(0, epochs)

#         # plot regression error
#         fig, axs = plt.subplots(figsize = (6,5))
#         sns.lineplot(x_axis, results['validation_0']['rmse'], label='Train', ax = axs)
#         sns.lineplot(x_axis, results['validation_1']['rmse'], label='Test', ax = axs)
#         plt.legend()
#         plt.ylabel('Regression Error')
#         plt.title('XGBoost Regression Error')
#         plt.show()

        return (pd.DataFrame(preds, columns = ['xgBoost_Predicted']),rmse, accur)
    
    def eNet_model(self,model):
        
        print("Begining to run ElasticNet Model.................................\n")
        print(f"Model used for Fitting and Predicting is {model}.....................\n")
        enet_grid  = model
        
        print("Fit train data to the best model.....................\n") 
        enet_grid.fit(self.X_train, self.y_train)
        
        print("Predict Y for test data.....................\n") 
        predictions = enet_grid.predict(self.X_test)
        
        RMSE = np.sqrt(mean_squared_error(self.y_test, predictions))
        r2 = enet_grid.score(self.X_test, self.y_test)
        #grid_results = pd.DataFrame(enet_grid.cv_results_) 

        print("ElasticNet Model report")
        print("----------------------------------------------------------------\n")
        
        print("Test RMSE: %f" % (RMSE))
        print('The accuracy of the elasticNet regressor is {:.2f} out of 1 on the test data'.format(r2))

        return (pd.DataFrame(predictions, columns = ['eNet_predicted']), RMSE, r2)
    
    def runModel(self, model_, whichModel = "LR"):
        if(whichModel == "LR"):
            return self.linearmodel(model_)
        elif(whichModel == "Lasso"):
            return self.classic_lasso_model(model_)
        elif(whichModel == "LassoCV"):
            return self.lasso_CV(model_)
        elif(whichModel == "eNet"):
            return self.eNet_model(model_)
        elif(whichModel == "RF"):
            return self.RF_model(model_)
        elif(whichModel == "xgb"):
            return self.xgb_model(model_)

    def get_y_test(self):
        return self.y_test


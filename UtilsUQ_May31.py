#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import libraries

import scipy.io as sio
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.preprocessing import StandardScaler,PolynomialFeatures,MinMaxScaler,RobustScaler
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from mapie.metrics import regression_coverage_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import explained_variance_score
from matplotlib.legend import _get_legend_handles_labels
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # disable debugging logs from Tensorflow
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from numpy import sqrt
from sklearn.datasets import fetch_california_housing
from scipy.stats import randint, uniform
from matplotlib.offsetbox import TextArea, AnnotationBbox
from matplotlib.ticker import FormatStrFormatter

#import libraries
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import scipy.io as sio
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from mapie.metrics import regression_coverage_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import explained_variance_score
from matplotlib.legend import _get_legend_handles_labels
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # disable debugging logs from Tensorflow
import warnings
from sklearn.preprocessing import StandardScaler
from numpy import sqrt
from sklearn.datasets import fetch_california_housing
from scipy.stats import randint, uniform
from matplotlib.offsetbox import TextArea, AnnotationBbox
from matplotlib.ticker import FormatStrFormatter

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from matplotlib.offsetbox import AnnotationBbox, TextArea
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import randint, uniform
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split

from mapie.metrics import (regression_coverage_score,
                           regression_mean_width_score)
from mapie.quantile_regression import MapieQuantileRegressor
from mapie.regression import MapieRegressor
from mapie.subsample import Subsample



random_state = 23
rng = np.random.default_rng(random_state)
round_to = 3

warnings.filterwarnings("ignore")
plt.rc('xtick',labelsize=19)
plt.rc('ytick',labelsize=19)
plt.rc('axes', labelsize=20, titlesize=16)
# To plot consistent and pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
# mpl.rc('axes', labelsize=14)
# mpl.rc('xtick', labelsize=12)
# mpl.rc('ytick', labelsize=12)
# mpl.rcParams['font.family'] = 'times new roman'
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['figure.dpi'] = 120
mpl.rcParams['savefig.dpi'] = 120
def coverage_score(Ytest, PIs_low, PIs_up):
    count = 0
    for i in range(len(Ytest)):
        # Check if the true value is within the prediction interval for each output
        if Ytest[i] >= PIs_low[i] and Ytest[i] <= PIs_up[i]:
            count += 1
    coverage = count / len(Ytest)
    return coverage
def CV_multi(X,Y,Xtest,Ytest,model,alpha,K_CV=100):
    n = X.shape[0]                           # Get the number of training samples
    m = int(n/K_CV)                          # Calculate the number of samples per fold
    ntest = Xtest.shape[0]                   # Get the number of test samples
    R = np.zeros((n, 2))                      # Initialize an array to store the residuals
    res=[]                                   # Initialize a list to store the results
    for k in np.arange(K_CV):                # Loop through the number of cross validation folds
        fold = np.arange(k*m,(k+1)*m)        # Get the indices of the current fold 
        X_ = np.delete(X,fold,0)             # Remove the current fold from the training data
        Y_ = np.delete(Y,fold,0)             # Remove the current fold from the training data
        model.fit(X_, Y_)              # Fit the model on the reduced training data
        # Calculate the absolute residuals for the current fold
        R[fold, :] = np.abs(Y[fold, :] - model.predict(X[fold, :]))
        
    PIs = np.zeros((ntest, 2, 2))              # Initialize an array to store the prediction intervals
    y_pred=np.zeros((ntest, 2))                 # Initialize an array to store the predictions
    model.fit(X, Y)                     # Fit the model on the entire training data
    y_pred=model.predict(Xtest)         # Make predictions on the test data
    for itest in np.arange(ntest):            # Loop through the test samples
        for i_output in range(2):
            # Calculate the lower bound of the prediction interval
            q_lo = np.sort(y_pred[itest, i_output]-R[:, i_output])[::-1][(np.ceil((1-alpha)*(n+1))).astype(int)]
            # Calculate the upper bound of the prediction interval
            q_hi = np.sort(y_pred[itest, i_output]+R[:, i_output])[(np.ceil((1-alpha)*(n+1))).astype(int)]
            # Store the lower and upper bounds in the prediction intervals array
            PIs[itest, i_output, :] = np.array([q_lo,q_hi])
    if Ytest is None:
        res.append(PIs)
    else:
    # Calculate the coverage of the prediction intervals
        PIs_low = PIs[:, 0, 0]
        PIs_up = PIs[:, 0, 1]
        coverage = np.zeros(Y.shape[1])
        width = np.zeros(Y.shape[1])
        for i_output in range(Y.shape[1]):
            coverage[i_output] = round(coverage_score(Ytest[:, i_output], PIs[:, i_output, 0], PIs[:, i_output, 1]),3)
            width[i_output] = (PIs[:, i_output, 1] - PIs[:, i_output, 0]).mean().round(3)

        res.append([coverage,width,PIs]) # Append the coverage, width, and prediction intervals to the results list
    return res
def CV_plus_multi(X,Y,Xtest,Ytest,model,alpha,K_CV=100):
    n = X.shape[0]
    m = int(n/K_CV)      
    ntest = Xtest.shape[0]
    y_pred=np.zeros((n,ntest,2))
    R = np.zeros((n,2))
    res=[]
    for k in np.arange(K_CV):
        fold = np.arange(k*m,(k+1)*m)    
        X_ = np.delete(X,fold,0)
        Y_ = np.delete(Y,fold,0)
        model.fit(X_, Y_)
        R[fold] = np.abs(Y[fold,:] -model.predict(X[fold,:]))
        y_pred[fold]=model.predict(Xtest)
    PIs = np.zeros((ntest,2,2))
    for itest in np.arange(ntest):
        for i_output in range(2):
            # Calculate the lower bound of the prediction interval
            q_lo = np.sort(y_pred[:,itest, i_output]-R[:, i_output])[::-1][(np.ceil((1-alpha)*(n+1))).astype(int)]
            # Calculate the upper bound of the prediction interval
            q_hi = np.sort(y_pred[:,itest, i_output]+R[:, i_output])[(np.ceil((1-alpha)*(n+1))).astype(int)]
            # Store the lower and upper bounds in the prediction intervals array
            PIs[itest, i_output, :] = np.array([q_lo,q_hi])
    if Ytest is None:
        res.append(PIs)
    else:

            
                # Calculate the coverage of the prediction intervals
        PIs_low = PIs[:, 0, 0]
        PIs_up = PIs[:, 0, 1]
        coverage = np.zeros(Y.shape[1])
        width = np.zeros(Y.shape[1])
        for i_output in range(Y.shape[1]):
            coverage[i_output] = round(coverage_score(Ytest[:, i_output], PIs[:, i_output, 0], PIs[:, i_output, 1]),3)
            width[i_output] = (PIs[:, i_output, 1] - PIs[:, i_output, 0]).mean().round(3)
        res.append([coverage,width,PIs])
    return res

def weighted_CV_plus_multi(X,Y,Xtest,Ytest,model,alpha,K_CV=100):
    n = X.shape[0]
    m = int(n/K_CV)      
    ntest = Xtest.shape[0]
    y_pred=np.zeros((n,ntest,2))
    R = np.zeros((n,2))
    d=np.zeros((n,ntest))
    w=np.zeros((n,ntest))
    res=[]
    for k in np.arange(K_CV):
        fold = np.arange(k*m,(k+1)*m)    
        X_ = np.delete(X,fold,0)
        Y_ = np.delete(Y,fold,0)
        model.fit(X_, Y_)
        R[fold] = np.abs(Y[fold,:] -model.predict(X[fold,:]))
        y_pred[fold]=model.predict(Xtest)
        d[fold]=1/(cdist(X[fold],Xtest,'euclidean'))
    PIs = np.zeros((ntest,2,2))
    for itest in np.arange(ntest):
        for i_output in range(2):
            w[:,itest] = d[:,itest]/d[:,itest].sum()
            q_lo = np.sort(y_pred[:,itest, i_output]-(w[:,itest]*R[:, i_output]*n))[::-1][(np.ceil((1-alpha)*(n+1))).astype(int)]
            q_hi = np.sort(y_pred[:,itest, i_output]+(w[:,itest]*R[:, i_output]*n))[(np.ceil((1-alpha)*(n+1))).astype(int)]
            PIs[itest, i_output,:] = np.array([q_lo,q_hi])
    # Calculate the coverage of the prediction intervals
    if Ytest is None:
        res.append(PIs)
    else:
        PIs_low = PIs[:, 0, 0]
        PIs_up = PIs[:, 0, 1]
        coverage = np.zeros(Y.shape[1])
        width = np.zeros(Y.shape[1])
        for i_output in range(Y.shape[1]):
            coverage[i_output] = round(coverage_score(Ytest[:, i_output], PIs[:, i_output, 0], PIs[:, i_output, 1]),3)
            width[i_output] = (PIs[:, i_output, 1] - PIs[:, i_output, 0]).mean().round(3)
        res.append([coverage,width,PIs])
    return res
def plot_PI(
    title,
    axs,
    y_test_sorted,
    y_pred_sorted,
    lower_bound,
    upper_bound,
    coverage,
    width,
    num_plots_idx
):
    """
    Plot of the prediction intervals
    """
    axs.yaxis.set_major_formatter(FormatStrFormatter('%.0f' ))
    axs.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    lower_bound_ = np.take(lower_bound, num_plots_idx)
    y_pred_sorted_ = np.take(y_pred_sorted, num_plots_idx)
    y_test_sorted_ = np.take(y_test_sorted, num_plots_idx)

    error = y_pred_sorted_-lower_bound_

    warning1 = y_test_sorted_ > y_pred_sorted_+error
    warning2 = y_test_sorted_ < y_pred_sorted_-error
    warnings = warning1 + warning2
    axs.errorbar(
        y_test_sorted_[~warnings],
        y_pred_sorted_[~warnings],
        yerr=error[~warnings],
        capsize=2, marker="o", elinewidth=1, linewidth=0,ecolor='lightgray',color='lightgray',
        label="Inside prediction interval"
        )
    axs.errorbar(
        y_test_sorted_[warnings],
        y_pred_sorted_[warnings],
        yerr=error[warnings],
        capsize=5, marker="o", elinewidth=2, linewidth=0, color="red",
        label="Outside prediction interval"
        )
    axs.scatter(
        y_test_sorted_[warnings],
        y_test_sorted_[warnings],
        s=40,marker="*", color="green",
        label="True value"
    )

    lims = [
        np.min([axs.get_xlim(), axs.get_ylim()]),  # min of both axes
        np.max([axs.get_xlim(), axs.get_ylim()]),  # max of both axes

    ]
    axs.plot(lims, lims, '--', alpha=0.75, color="black", label="x=y")
    axs.set_title(title+  ("(%.2f, %.0f)" % (coverage, width)),fontweight ='bold', fontsize = 25)

    axs.set( adjustable='box',aspect='equal') #adjustable='box',

# main function 
def TrainModel(file_name, scaler='std',method='WCV+',confidence=0.95,test_size=0.2, frac=1):
    starttime=time.time()
    data = pd.read_csv(file_name)
    print("The orange line within each IQR box represents the median value, " 
      "while the green diamond represents the mean. Outliers are shown as red circles. " 
      "Let's look at the boxplots of inputs:")

    columns = ['Vf_IQR', 'Vf_Mdn', 'nFC', 'nMRC', 'FCp', 'MRCp', 'av_FCa', 'av_MRCa', 'max_FC', 'max_MRC']
    red_circle = dict(markerfacecolor='red', marker='o')
    mean_shape = dict(markerfacecolor='green', marker='D', markeredgecolor='green')
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))

    for i, col in enumerate(columns):
        ax = axs[i//5, i%5]  # Get the appropriate axis for the current column
        ax.boxplot(data[col], vert=False,flierprops=red_circle,showmeans=True, meanprops=mean_shape)
        ax.set_title(col, fontsize=16, fontweight='bold')  # Set the title for the axis
        ax.set_yticklabels([])  # Hide the y-axis labels
        ax.grid(False)
        # Set the font size and weight for the x-axis and y-axis labels
        for label in ax.get_xticklabels() + ax.get_yticklabels() :
            label.set_fontsize(12)
            label.set_weight('bold')
    plt.tight_layout()
    plt.show()
    columns = ['Stiffness', 'Strength']

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))
    print("Let's look at the boxplots of targets:")
    for i, col in enumerate(columns):
        ax = axs[i]  # Get the appropriate axis for the current column
        ax.boxplot(data[col], vert=False,flierprops=red_circle,showmeans=True, meanprops=mean_shape)
        ax.set_title(col, fontsize=12, fontweight='bold')  # Set the title for the axis
        ax.set_yticklabels([])  # Hide the y-axis labels

        # Set the font size and weight for the x-axis and y-axis labels
        for label in ax.get_xticklabels() + ax.get_yticklabels() :
            label.set_fontsize(10)
            label.set_weight('bold')
#     axs[0].set_ylabel("original data", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    X=data.iloc[:,2:12].to_numpy()
    y=data[['Stiffness','Strength']].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    scalers={'std':StandardScaler,'minmax':MinMaxScaler, 'robust':RobustScaler}
    scaler=scalers[scaler]()
#     scaler = MinMaxScaler() #MinMaxScaler() StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    estimator = MultiOutputRegressor(LGBMRegressor(
        objective='quantile',
        alpha=0.5,
        random_state=random_state
    ))

    params_distributions = {
        'estimator__num_leaves': randint(low=10, high=50),
        'estimator__max_depth': randint(low=3, high=20),
        'estimator__n_estimators': randint(low=50, high=300),
        'estimator__learning_rate': uniform()
    }

    optim_model = RandomizedSearchCV(
        estimator,
        param_distributions=params_distributions,
        n_jobs=-1,
        n_iter=100,
        cv=KFold(n_splits=5, shuffle=True),
        verbose=-1
    )
    optim_model.fit(X_train, y_train)
    estimator = optim_model.best_estimator_
    methods = {
        'WCV+': weighted_CV_plus_multi,
        'CV+': CV_plus_multi,
        'CV': CV_multi
    }
    results = methods[method](X_train, y_train, X_test, y_test, estimator, alpha=1-confidence)
     ##########Stiffness#########
    a1=pd.DataFrame(list(zip(y_test[:,0],results[0][2][:, 0, :][:,0], results[0][2][:, 0, :][:,1],)),
                   columns=['Stiffness','low','up', ])
    ##########Strength#########
    a2=pd.DataFrame(list(zip(y_test[:,1],results[0][2][:, 1, :][:,0], results[0][2][:, 1, :][:,1],)),
                   columns=['Strength','low','up',])   

    a1_sort=a1.sort_values(by=['Stiffness'])
    a2_sort=a2.sort_values(by=['Strength'])
    y_test1=a1_sort.Stiffness.values
    
    r1={}
    r1[method]=np.array(a1_sort[['low','up']])
    coverage1={}
    width1={}
    coverage1[method] = round(coverage_score(y_test1,r1[method][:,0], r1[method][:,1]),3)
    width1[method]=(r1[method][:,1] - r1[method][:,0]).mean().round(3)
    y_test2=a2_sort.Strength.values
    r2={}
    r2[method]=np.array(a2_sort[['low','up']])
    coverage2={}
    width2={}
    coverage2[method] = round(coverage_score( y_test2,r2[method][:,0], r2[method][:,1]),3)
    width2[method]=(r2[method][:,1] - r2[method][:,0]).mean().round(3)
    print("Confidence level is set at %s percent!\nThe size of your test set is %s!"%(confidence*100,y_test.shape[0]))
    print("The method used for uncertainty quantification is %s" %method)
    print("The first element in the paranthesis refers to the fraction of test points whose true values are within the PIs.")
    print("The second element in the paranthesis refers to the average of PI widths over all samples in the test set.")   
    print("Stiffness: %.2f percent of test set falls within PIs while the average PI is %s "%((coverage1[method]*100, width1[method])))
    print("Strength: %.2f percent of test set falls within PIs while the average PI is %s "%((coverage2[method]*100, width2[method])))
    print("Running time: It takes %0.2f seconds to train the model!"%(time.time()-starttime))
    perc_obs_plot = frac
    num_plots = rng.choice(
        len(y_test), int(perc_obs_plot*len(y_test)), replace=False
        )
    fig, axs = plt.subplots(1, 2, figsize=(15,9)) 
    coords = axs[0] , axs[1]

    plot_PI(
            method,
            axs[0],
            y_test1,
            r1[method].mean(axis=1).ravel(),
            r1[method][:,0],
            r1[method][:,1],
            coverage1[method],
            width1[method],
            num_plots
            )
    plot_PI(
            method,
            axs[1],
            y_test2,
            r2[method].mean(axis=1).ravel(),
            r2[method][:,0],
            r2[method][:,1],
            coverage2[method],
            width2[method],
            num_plots
            )
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(_, []) for _ in zip(*lines_labels)]
    plt.legend(
        lines[:4], labels[:4],
        loc='upper center',
        bbox_to_anchor=(-0.02,-0.15 ), 
        fancybox=True,
        shadow=True,
        ncol=4,fontsize = 18,
    )


    axs[0].set_ylabel("Prediction Interval",fontweight ='bold', fontsize = 20)
    axs[0].set_xlabel("True Stiffness",fontweight ='bold', fontsize = 20) 
    axs[1].set_xlabel("True Strength",fontweight ='bold', fontsize = 20)
    plt.show()
    # Create DataFrames for X_test and y_test
    X_test_df = pd.DataFrame(X_test)
    y_test_df = pd.DataFrame(y_test)

    # Concatenate the two DataFrames column-wise
    result = pd.concat([X_test_df, y_test_df], axis=1)

    result.columns = ['Vf_IQR', 'Vf_Mdn', 'nFC', 'nMRC', 'FCp', 'MRCp', 'av_FCa', 'av_MRCa', 'max_FC', 'max_MRC','Stiffness', 'Strength']
    result['Stiffness_lower']=results[0][2][:, 0, :][:,0]
    result['Stiffness_upper']=results[0][2][:, 0, :][:,1]
    result['Strength_lower']=results[0][2][:, 1, :][:,0]
    result['Strength_upper']=results[0][2][:, 1, :][:,1]
    
def Solution(file_name, want,method='WCV+',confidence=0.95):
    starttime=time.time()
    data = pd.read_csv(file_name)
    want = pd.read_csv(want)
    X_train=data.iloc[:,2:12].to_numpy()
    y_train=data[['Stiffness','Strength']].to_numpy()
    X_test=want.iloc[:,0:10].to_numpy()
    #Scaling Transform
#     scaler = StandardScaler()
#     X_train=scaler.fit_transform(X_train)
#     X_test=scaler.transform(X_test)

    estimator = MultiOutputRegressor(LGBMRegressor(
        objective='quantile',
        alpha=0.5,
        random_state=random_state
    ))
    params_distributions = {
        'estimator__num_leaves': randint(low=10, high=50),
        'estimator__max_depth': randint(low=3, high=20),
        'estimator__n_estimators': randint(low=50, high=300),
        'estimator__learning_rate': uniform()
    }
    optim_model = RandomizedSearchCV(
        estimator,
        param_distributions=params_distributions,
        n_jobs=-1,
        n_iter=100,
        cv=KFold(n_splits=5, shuffle=True),
        verbose=-1
    )
    optim_model.fit(X_train, y_train)
    estimator = optim_model.best_estimator_
    methods = {
        'WCV+': weighted_CV_plus_multi,
        'CV+': CV_plus_multi,
        'CV': CV_multi
    }
    results = methods[method](X_train, y_train, X_test,Ytest=None, model=estimator, alpha=1-confidence)

    want['Stiffness_lower']=results[0][:, 0, :][:,0]
    want['Stiffness_upper']=results[0][:, 0, :][:,1]
    want['Strength_lower']=results[0][:, 1, :][:,0]
    want['Strength_upper']=results[0][:, 1, :][:,1]
    print("Running time: It takes %0.2f seconds to get the solution."%(time.time()-starttime))

    return want


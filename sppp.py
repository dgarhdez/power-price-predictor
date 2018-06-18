
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import requests
import datetime
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.isotonic import IsotonicRegression

from scipy.interpolate import interp1d
import statsmodels.api as sm

def dateFormatting(iniDate, endDate):
    """
    Converts a pair of dates (YYYYMMDD, YYYYMMDD) into (YYYY-MM-DD 00:00:00, YYYY-MM-DD 23:59:59)
    
    :iniDate: first date of the interval, 'YYYYMMDD'
    :endDate: last date of the interval, 'YYYYMMDD'
    """
    iniDate = datetime.datetime.strftime(datetime.datetime.strptime(iniDate,'%Y%m%d'), '%Y-%m-%d %H:%M:%S')
    endDate = datetime.datetime.strptime(endDate, '%Y%m%d') # convert to datetime
    endDate = endDate + datetime.timedelta(days=1) - datetime.timedelta(seconds=1) # adds 1d and substracts 1s to get 23:59:59
    endDate = datetime.datetime.strftime(endDate, '%Y-%m-%d %H:%M:%S')  
    
    return iniDate,endDate

def listDates(endRange, days):
    """
    Returns a list containing all the dates between the two chosen.
    
    :endDate: last date of the range of dates
    :days: number of days included in the list
    """
    d1 = datetime.datetime.strptime(endRange, '%Y%m%d') - datetime.timedelta(days=days)  # start date
    d2 = datetime.datetime.strptime(endRange, '%Y%m%d')  # end date

    delta = d2 - d1         # timedelta
    lstDates = []
    for i in range(delta.days + 1):
        dt = datetime.datetime.strftime(d1 + datetime.timedelta(i), '%Y%m%d')
        lstDates.append(dt)
    return lstDates
    
def indicatorRequest(iniDate, endDate, indicator):
    """
    Basic function to fetch data from api.esios.ree.es
    Using requests - http://docs.python-requests.org/en/master/
    You need to ask for a token - consultasios@ree.es
    
    The indicators avilable are included in 'indicadores.csv', including id and name
    
    Requires a certain datetime format - YYYY-MM-DD HH:MM:SS, that's why dateFormatting is used for.
    Returns a pandas dataframe which includes the indicator in the input and datetime as index
    
    :iniDate: first date of the interval, 'YYYYMMDD'
    :endDate: last date of the interval, 'YYYYMMDD'  
    :indicator: integer, the id in 'indicadores.csv', e.g. 541 for Wind Power Forecast
    """
    token = '<INSERT TOKEN HERE>'
    
    iniDate, endDate = dateFormatting(iniDate,endDate) # transforms dates for request
    
    header = {'Accept': 'application/json; application/vnd.esios-api-v1+json',
              'Content-Type': 'application/json',
              'Host': 'api.esios.ree.es',
              'Authorization': 'Token token="{0}"'.format(token),
              'Cookie':'' }
    
    param = {'start_date':iniDate, 'end_date':endDate, 'time_trunc':'hour'}
    url = 'https://api.esios.ree.es/indicators/' + str(indicator)
    r = requests.get(url, headers=header,params=param).json()
    
    # json to dataframe
    vals = r['indicator']['values']
    df = pd.DataFrame(vals) 
    
    if 3 in df['geo_id'].unique(): # For indicator 600 - Spain power price
        df = df[df['geo_id'] == 3]

    df['dt'] = [x[:10] + ' ' + x[11:13] + ':00:00' for x in df['datetime']]       
    df = df.set_index(pd.to_datetime(df['dt']))[['value']]        
    df.columns = [str(indicator)]
    
    if indicator == 474: # For indicator 474 - nuclear availability
        return df.groupby('dt').sum()
    else:
        return df

def datasetBatch(iniDate, endDate, indList):
    """
    Returns a dataset including the values of every indicator included in indList
    
    :iniDate: first date of the interval, 'YYYYMMDD'
    :endDate: last date of the interval, 'YYYYMMDD'  
    :indList: list of integer, e.g. [541, 460]
    """
    
    df = pd.DataFrame()
    
    for ind in indList:
        req = indicatorRequest(iniDate, endDate, ind)
        df = pd.concat([df, req], axis=1)
        
    return df

def datasets(iniDate, endDate, fcDate):
    """
    Creates training and evaluation datasets, including all the indicators in indList
    Wrangling: engineered features - HT, wtdRatio, drtdRatio - to improve prediction
    
    In the code there's included the indicators I use - indList
    
    :iniDate: first date of the interval, 'YYYYMMDD'
    :endDate: last date of the interval, 'YYYYMMDD'
    :fcDate: date of the day to be forecasted, should be greater than endDate
    """
    
    month = int(fcDate[4:6])
    
    indList = [10249, 474, 541, 460, 600]
    
    trainData = datasetBatch(iniDate, endDate, indList)
    evalData = datasetBatch(fcDate, fcDate, indList[:-1]) # Everything but the target - 600
    
    for df in [trainData, evalData]:
        df['HT'] = df['10249'] - df['474'] # Residual demand - Nuclear availability
        df['wtdRatio'] = df['541'] / df['460'] # Wind power / Demand
        df['drtdRatio'] = df['10249'] / df['460'] # Residual demand / Demand
        df['hour'] = [x.hour for x in df.index]
        df['month'] = [x.month for x in df.index]
        df['weekday'] = [x.weekday() for x in df.index]
        df['weekend'] = [1 if x.weekday()>4 else 0 for x in df.index]
 
    
    return trainData, evalData
    
def linearPredict(trainData, evalData):
    """
    Based on http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    
    Fits linear regression (PMD ~ HT) between iniDate and endDate and then predicts values for fcDate.
    Requires training and evaluation datasets to be input
    
    Train-test split with a test size of 25 %
    
    :trainData: pandas dataframe
    :evalData: pandas dataframe
    """
    algo = 'linear'
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(trainData['HT'].values.reshape(-1, 1),
                                                            trainData['600'].values.reshape(-1, 1),
                                                            test_size=.25, random_state=42)
        # regression    
        linear = LinearRegression()
        linear.fit(X_train, y_train)
        testPred = linear.predict(X_test)
        r2_test = r2_score(y_test, testPred)
        linearPred = linear.predict(evalData['HT'].values.reshape(-1, 1))

        # results
        dicResults = {'r2':r2_test , 'pred':linearPred.flatten(), 'name':algo}

        return dicResults
    except ValueError:
        print('Error running "{0}". Resuming next calculations.'.format(algo))
        
def isotonicPredict(trainData, evalData):
    """
    Based on http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    
    Fits linear regression (PMD ~ HT) between iniDate and endDate and then predicts values for fcDate.
    Requires training and evaluation datasets to be input
    
    Train-test split with a test size of 25 %
    
    :trainData: pandas dataframe
    :evalData: pandas dataframe
    """
    algo = 'isotonic'
    
    #try:
    X_train, X_test, y_train, y_test = train_test_split(trainData['HT'].values.reshape(-1, 1),
                                                        trainData['600'].values.reshape(-1, 1),
                                                        test_size=.25, random_state=42)

    # regression    
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(X_train.flatten(), y_train.flatten())
    testPred = iso.predict(X_test.flatten())
    r2_test = r2_score(y_test, testPred)
    isoPred = iso.predict(evalData['HT'].values.reshape(-1, 1).flatten())

    # results
    dicResults = {'r2':r2_test , 'pred':isoPred.flatten(), 'name':algo}

    return dicResults
    #except ValueError:
        #print('Error running "{0}". Resuming next calculations.'.format(algo))

def loessPredict(trainData, evalData):
    """
    Based on http://www.statsmodels.org/dev/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html
    
    Fits loess regression (PMD ~ HT) between iniDate and endDate and then predicts values for fcDate.
    Requires training and evaluation datasets to be input
    
    Train-test split with a test size of 25 %
    
    Runs a GridSearch-ish optimization for the 'frac' parameter, choosing the one that achieves best R2 score
    
    :trainData: pandas dataframe
    :evalData: pandas dataframe  
    """
    algo = 'loess'
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(trainData['HT'].values.reshape(-1, 1),
                                                            trainData['600'].values.reshape(-1, 1),
                                                            test_size=.25, random_state=42)

        best = 0
        for frac in [.1,.2,.3,.4,.5,.6,.7,.8,.9]: # for choosing the best 'frac' 
            # lowess will return our "smoothed" data with a y value for at every x-value
            lowess = sm.nonparametric.lowess(y_train.flatten(), X_train.flatten(), frac=frac)

            # unpack the lowess smoothed points to their values
            lowess_x = list(zip(*lowess))[0]
            lowess_y = list(zip(*lowess))[1]

            # run scipy's interpolation
            loess = interp1d(lowess_x, lowess_y, bounds_error=False)

            # regression    
            dfTest = pd.DataFrame()
            dfTest['Xtest'] = X_test.flatten()
            dfTest['ytest'] = y_test.flatten()
            testPred = loess(X_test.flatten())
            dfTest['testPred'] = testPred
            dfTest = dfTest.dropna(how='any')
            r2_test = r2_score(dfTest['ytest'], dfTest['testPred'])
            loessPred = loess(evalData['HT'].values)
            if r2_test>best:
                d={}
                d['pred'] = loessPred
                d['r2'] = r2_test
                d['frac'] = frac
                best=r2_test

        # results
        dicResults = {'r2':d['r2'] , 'pred':d['pred'].flatten(), 'name':algo}

        return dicResults
    except ValueError:
        print('Error running "{0}". Resuming next calculations.'.format(algo))

def multiLinearPredict(trainData, evalData):
    """
    Based on http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    
    Fits multiple linear regression (PMD ~ 10249 + 474 + 541 + 460 + 600 + HT + wtdRatio + drtdRatio) 
        between iniDate and endDate and then predicts values for fcDate.
    Requires training and evaluation datasets to be input
    
    Train-test split with a test size of 25 %
    
    :trainData: pandas dataframe
    :evalData: pandas dataframe   
    """
    algo = 'multiLinear'
    
    try:
        XT = trainData.drop('600', axis=1)
        yT = trainData[['600']]

        # train-test split
        X_train, X_test, y_train, y_test = train_test_split(XT,yT,test_size=.25, random_state=42)

        # fitting and prediction
        multiLinear = LinearRegression()
        multiLinear.fit(X_train, y_train.values.ravel())
        testPred = multiLinear.predict(X_test)
        r2_test = r2_score(y_test, testPred)
        multiLinearPred = multiLinear.predict(evalData)

        # results
        dicResults = {'r2':r2_test , 'pred':multiLinearPred.flatten(), 'name':algo}

        return dicResults
    except ValueError:
        print('Error running "{0}". Resuming next calculations.'.format(algo))

def polyPredict(trainData, evalData, deg):    
    """
    Based on https://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html
    
    Fits a 'deg' degree polynomial (PMD ~ HT) between iniDate and endDate and then predicts values for fcDate.
    Requires training and evaluation datasets to be input
    
    Train-test split with a test size of 25 %
    
    :trainData: pandas dataframe
    :evalData: pandas dataframe  
    :deg: Degree of the fitting polynomial 
    """
    algo = 'poly'
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(trainData['HT'].values.reshape(-1, 1),
                                                            trainData['600'].values.reshape(-1, 1),
                                                            test_size=.25, random_state=42)

        # fitting and prediction
        fitter = np.polyfit(X_train.flatten(), y_train.flatten(), deg)
        polyfit = np.poly1d(fitter)
        testPred = polyfit(X_test.flatten())
        r2_test = r2_score(y_test.flatten(),testPred)
        polyPred = polyfit(evalData['HT'].values)

        # results
        dicResults = {'r2':r2_test , 'pred':polyPred.flatten(), 'name':algo}

        return dicResults
    except ValueError:
        print('Error running "{0}". Resuming next calculations.'.format(algo))

def knnPredict(trainData, evalData, kn):
    """
    Based on http://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-regression
    
    Performs k-Nearest Neighbors regression between iniDate and endDate and then predicts values for fcDate.
    Requires training and evaluation datasets to be input
    
    Train-test split with a test size of 25 %
    
    :trainData: pandas dataframe
    :evalData: pandas dataframe  
    :kn: Number of neighbors to use 
    """
    algo = 'knn'
    
    try:    
        XT = trainData.drop('600', axis=1)
        yT = trainData[['600']]

        # train-test split
        X_train, X_test, y_train, y_test = train_test_split(XT,yT,test_size=.25, random_state=42)

        # fitting and prediction
        knn = KNeighborsRegressor(n_neighbors = kn)
        knn.fit(X_train, y_train.values.ravel())
        testPred = knn.predict(X_test)
        r2_test = r2_score(y_test, testPred)
        knnPred = knn.predict(evalData)

        # results
        dicResults = {'r2':r2_test , 'pred':knnPred.flatten(), 'name':'knn'}

        return dicResults
    except ValueError:
        print('Error running "{0}". Resuming next calculations.'.format(algo))

def rforestPredict(trainData, evalData, trees):
    """
    Based on http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    
    Performs a Random Forest Regression between iniDate and endDate and then predicts values for fcDate.
    Requires training and evaluation datasets to be input
    
    Train-test split with a test size of 25 %
    
    :trainData: pandas dataframe
    :evalData: pandas dataframe  
    :trees: The number of trees in the forest 
    """
    algo = 'rforest'
    
    try:
        XT = trainData.drop('600', axis=1)
        yT = trainData[['600']]

        # train-test split
        X_train, X_test, y_train, y_test = train_test_split(XT,yT,test_size=.25, random_state=42)

        # fitting and prediction
        rforest = RandomForestRegressor(n_estimators=trees, oob_score=True)
        rforest.fit(X_train, y_train.values.ravel())
        testPred = rforest.predict(X_test)
        r2_test = r2_score(y_test, testPred)
        rforestPred = rforest.predict(evalData)

        # results
        dicResults = {'r2':r2_test , 'pred':rforestPred.flatten(), 'name':'rforest'}

        return dicResults
    except ValueError:
        print('Error running "{0}". Resuming next calculations.'.format(algo))

def gbrPredict(trainData, evalData, trees):
    """
    Based on http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    
    Performs a Random Forest Regression between iniDate and endDate and then predicts values for fcDate.
    Requires training and evaluation datasets to be input
    
    Train-test split with a test size of 25 %
    
    :trainData: pandas dataframe
    :evalData: pandas dataframe  
    :trees: The number of trees in the forest 
    """
    algo = 'gbr'
    
    try:
        XT = trainData.drop('600', axis=1)
        yT = trainData[['600']]

        # train-test split
        X_train, X_test, y_train, y_test = train_test_split(XT,yT,test_size=.25, random_state=42)

        # fitting and prediction
        p = {'learning_rate': 0.1, 'max_depth': 5, 'subsample': 0.8}
        gbr = GradientBoostingRegressor(n_estimators=trees, **p)

        gbr.fit(X_train, y_train.values.ravel())
        testPred = gbr.predict(X_test)
        r2_test = r2_score(y_test, testPred)
        gbrPred = gbr.predict(evalData)

        # results
        dicResults = {'r2':r2_test , 'pred':gbrPred.flatten(), 'name':algo}

        return dicResults
    except ValueError:
        print('Error running "{0}". Resuming next calculations.'.format(algo))

def forecasting(trainData, evalData):
    """
    Wisdom of the crowd
    Ensembling of the different predictions from the following methods: linear regression, isotonic regression, loess,
      polynomial, multilinear, k-Neighbors, Random Forest and Gradient Boosting
     
    :trainData: pandas dataframe
    :evalData: pandas dataframe       
    """
    
    linear = linearPredict(trainData, evalData)
    isotonic = isotonicPredict(trainData, evalData)
    loess = loessPredict(trainData, evalData)
    polyReg = polyPredict(trainData, evalData, 3)
    multiLinear = multiLinearPredict(trainData, evalData)
    knn = knnPredict(trainData, evalData, 15)
    rforest = rforestPredict(trainData, evalData, 1000)
    gbr = gbrPredict(trainData, evalData, 1000)
    total = [loess, polyReg, multiLinear, knn, rforest, gbr]

    preds = pd.DataFrame()  
    r2 = pd.DataFrame()
    nombres = []
    r2list = []
    for item in total:
        preds[item['name']] = item['pred']
        nombres.append(item['name'])
        r2list.append(item['r2'])
    
    # preds
    preds = preds.set_index(evalData.index)
    
    # r2
    r2['algoritmo'] = nombres
    r2['r2'] = r2list
    
    return  preds, r2

def results(preds, dia):
    preds['avg'] = preds.mean(axis=1)
    print('avgModels | avg = {0:.2f} €/MWh'.format(preds['avg'].mean()))

    # predictions column
    x = [x.hour for x in preds.index]
    for column in preds:   
        plt.plot(x , preds[column], marker='', color='grey', linewidth=1, alpha=0.4)

    # highlight average
    plt.plot(x, preds['avg'], marker='', color='green', linewidth=3, alpha=0.7)
    plt.text(23, preds['avg'][23], ' AVG', horizontalalignment='left', size='small', color='green', fontsize=14)    

    # when day-ahead day is published
    try: 
        preds['pmd'] = indicatorRequest(dia, dia, 600)['600'].values
        plt.plot(x, preds['pmd'], marker='.', color='orangered', linewidth=4, alpha=0.7)
        plt.text(23, preds['pmd'][23], ' PMD', horizontalalignment='left', size='small', color='orangered', fontsize=14)
        print('Price     | PMD = {0:.2f} €/MWh'.format(preds['pmd'].mean()))
    except KeyError:
        print('Price not published yet for day {0}'.format(dia))

    # Add titles
    plt.margins(xmargin=0.15)
    plt.xlabel("Hora")
    plt.ylabel("Precio")
    plt.show()
    
def horizontal_avgprice_line(x, **kwargs):
    plt.axhline(x.mean(), linestyle =':', color = 'red')
    plt.text(.5,x.mean()+1, 'avg={0:.2f}\nstd={1:.2f}'.format(x.mean(), x.std()), fontsize=11, color = 'red')
    
def vertical_avgprice_line(x, **kwargs):
    plt.axvline(x.mean(), linestyle =':', color = 'red')
    
def pmd(days):
    """
    Final function to use
    Displays graph of the average of predictions and the mean price for the day-ahead.
    
    :days: length of the training dataset to predict tomorrow's price
    """
    
    dia = datetime.datetime.strftime(datetime.datetime.today().date() + datetime.timedelta(days=1), '%Y%m%d')
    fin = datetime.datetime.strftime(datetime.datetime.today().date(), '%Y%m%d')
    ini = datetime.datetime.strftime(datetime.datetime.today().date() - datetime.timedelta(days=days), '%Y%m%d')

    # Creation of the training and evaluation datasets
    trainData, evalData = datasets(ini, fin, dia)

    # Forecasting
    preds, r2 = forecasting(trainData, evalData)

    # Plotting results
    graphs = results(preds, dia)
    
    return preds

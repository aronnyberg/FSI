#Imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn import ensemble
import pickle

#Sort data
data = pd.read_excel('your excel file here')
data.rename(columns={'Unnamed: 0':'Date'}, inplace = True)
data = data.sort_values('Date')
data.set_index('Date', inplace=True)
logRets = np.log(((data / data.shift(+1)) -1)+1)
Z = data['Asset1']
logZ = logRets['Asset1']
Zmonthly = data.asfreq('M').ffill().dropna()
logRetsMonthly = np.log(((Zmonthly / Zmonthly.shift(+1)) -1)+1)

#Compute Markowitz function
def markowitz(stock_list, log_returns):
    np.random.seed(42)
    num_ports = 100 #alter iterations searched
    all_weights = np.zeros((num_ports, len(stock_list)))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)
    
    for x in range(num_ports):
        # Weights
        #Potential to add leverage here
        weights = np.array(np.random.random(len(stock_list)))
        weights = weights/np.sum(weights)
        # Save weights
        all_weights[x,:] = weights
        # Expected return
        ret_arr[x] = np.sum( (log_returns.mean() * weights * 252))
        # Expected volatility
        vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov()*252, weights)))
        # Expected Sharpe Ratio
        sharpe_arr[x] = ret_arr[x]/vol_arr[x]
        weights = np.round(all_weights[sharpe_arr.argmax(),:], 3)
    return weights

#Run Markowitz through time
#days = window length, data=your log returns, fees=trading cost
def moving_markowitz(days, data, fees):
    X = data
    n_train = days*10
    n_records = len(X)
    weight_list = []
    returns_list = []
    date_list = []
    #Recomputes Markowitz weights based on all data up to that date
    for i in range(n_train, n_records, days):
        date = X.index.values[i]
        train = X[0:i]
        weight_list.append(markowitz(train.columns, train))
        returns = sum(sum(np.array(X[i:i+days])*weight_list[-1]))-fees
        returns_list.append(returns) 
        date_list.append(date)
    return pd.DataFrame({'Date':date_list, 'Returns':returns_list})

#Compute Momentum data
adict = {'Date':logRetsMonthly.index}
for i in logRetsMonthly:
    ma20 = logRetsMonthly[i].rolling(20).mean().values
    ma60 = logRetsMonthly[i].rolling(60).mean().values
    ma_diff = ma20-ma60
    adict.update({i:ma_diff})
ma = pd.DataFrame(adict).set_index('Date')

#Compute Momentum coefficents, operated within MA_ML_moving
def MA_ML(an_array, train_upto):
    an_array = an_array.reshape(-1,1)
    adict = {}
    tscv = TimeSeriesSplit(n_splits=5)
    trainX = ma.dropna().iloc[:train_upto,:]
    trainy = logRetsMonthly.shift(-1).dropna().iloc[:train_upto,:]
    #Computes random forest regressor predicted values, used as portfolio weightings
    for i in logRetsMonthly.columns.values:
        
        X = trainX[i]
        y = trainy[i]
        for train_index, test_index in tscv.split(X):
            
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            
            X_train = np.array(X_train).reshape(-1, 1)
            X_test = np.array(X_test).reshape(-1, 1)
            y_train = np.array(y_train).reshape(-1, 1)
            y_test = np.array(y_test).reshape(-1, 1)
            
        rf = ensemble.RandomForestRegressor()
        rf.fit(np.array(X_train), np.array(y_train))
        
    prediction = rf.predict(an_array)
    return prediction
#Run Momentum strategy
#assets = your log returns data
def MA_ML_moving(assets, trade_cost):
    #X = ma.dropna()
    returns = assets.dropna()
    n_train = 100
    weight_list = []
    returns_list = []
    date_list = []
    strat_returns_list = []
    #Recomputes Random forest regressor on all data up to date
    for i in range(n_train, len(returns)):
        date = returns.index.values[i]
        date_before = returns.index.values[i-1]
        weight_list.append(MA_ML(ma.loc[date].dropna().values, int(ma.reset_index()[ma.index == returns.index.values[i]].index[0])-1))
        allocation = sum(weight_list)
        strat_returns = sum(sum(np.array(returns.iloc[i,:].values)*weight_list[-1])-(allocation*trade_cost))
        strat_returns_list.append(strat_returns) 
        date_list.append(date)
    results = pd.DataFrame({'Date':date_list, 'MA_ML_Returns':strat_returns_list})
    results.set_index('Date', inplace=True)
    return results

#Compute and run Mean Reversion strategy
def MR():
    return_list = []
    index_list = []
    date_list = []
    #Allocates evenly for t+1 between worst two strategies in t
    for i in range(0, len(logRetsMonthly)-1):
        trades = 2
        date = logRetsMonthly.index.values[i]
        sorts = logRetsMonthly.iloc[i,:].sort_values(ascending=True).index.values
        #top = sorts[:2]
        bottom = sorts[-trades:]
        #top_returns = np.mean(logRetsMonthly[top].values[i+1])
        bottom_returns = np.mean(logRetsMonthly[bottom].values[i+1])
        #therefore allocates 100% and receives mean of trades
        strat_returns = bottom_returns
        return_list.append(strat_returns-(trades*0.001))
        date_list.append(date)
    df = pd.DataFrame({'MR_Returns':return_list, 'Date':date_list}).dropna().set_index('Date')
    return df
#Save and run the strategies
marko = moving_markowitz(1, logRetsMonthly, 0.001)
ma_ml = MA_ML_moving()
mr = MR()

#Convexity Plotter
def get_payoff(x, y):
    
    a = x
    b = y
    r = np.polyfit(a, b, 1)
    r2 = np.polyfit(a, b, 2)
    plt.plot(a, b, 'o')
    plt.plot(a,np.polyval(r, a), 'r-')
    plt.plot(a,np.polyval(r2, a), 'b--')

#Compute cVar
def VaR(asset, alpha):
    mu_norm, sig_norm = norm.fit(asset)
    dx = 0.0001  # resolution
    x = np.arange(-0.1, 0.1, dx)
    pdf = norm.pdf(x, mu_norm, sig_norm)
    parm = t.fit(asset)
    nu, mu_t, sig_t = parm
    nu = np.round(nu)
    pdf2 = t.pdf(x, nu, mu_t, sig_t)
    h = 1
    lev = 100*(1-alpha)
    xanu = t.ppf(alpha, nu)
    VaR_t = np.sqrt((nu-2)/nu) * t.ppf(1-alpha, nu)*sig_norm  - h*mu_norm
    CVaR_t = -1/alpha * (1-nu)**(-1) * (nu-2+xanu**2) * \
                t.pdf(xanu, nu)*sig_norm  - h*mu_norm
    return CVaR_t

#Join strategy returns
all_df = pd.concat([mr, marko, ma_ml],axis=1)
all_df2 = all_df.dropna()
#all_df_exml are sub-strategies ex momentum strategy
all_df_exml = pd.concat([mr, marko],axis=1)
all_df2_exml = all_df_exml.dropna()
#And compute portfolio weightings through time
all_reb_exml = moving_markowitz(1, all_df2_exml, 0)
#Strategy returns, sharpe ratio and drawdowns
returns = all_reb_exml
cumReturns = 1+all_reb_exml_unc.cumsum()
#dd=Absolute level of drawdowns, dd2= 'actual' drawdowns through time
dd = cumReturns.rolling(100000, min_periods = 1).max()
dd2 = (cumReturns/dd - 1)
sharpe = returns.mean()/returns.std()
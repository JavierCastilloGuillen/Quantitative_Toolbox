import pandas as pd
import numpy as np
import statsmodels.tsa.api as smt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, arma_order_select_ic
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")
sns.set_theme()


# Helper Functions in case are needed for the data


def Diff(x):
    x_diff = 100*(x/x.shift(1))
    return  x_diff - 100

def rmse(predicted, sample):
    from math import sqrt
    """ Function to get Root Squared Mean Error.
    Args:
        pred (array): Predicted Values for Model
        target (array): Test values
    Returns:
        Float: Root Mean Squared Error
    """
    return sqrt(((predicted - sample) ** 2).mean())

def adf_test(timeseries):
    '''
    Augmented Dickey-Fuller Test for unit root testing over
    univariate series.
    '''
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    return dfoutput

def get_order_sbic(max_range, y_train):
    '''
    Iteration to retreive the minimum SBIC value to select order 
    for a model since statsmodels didn't work once I've tried
    order = get_order_sbic(5, y)[0][0] # Get the order
    After Statsmodels has been restored it is not necessary this function
    '''

    import itertools
    bic = []
    parameter = []
    p=d=q=range(0, max_range)

    pdq = list(itertools.product(p,d,q))

    for param in pdq:
        try:
            model_arima= ARIMA(y_train, order = param) 
            model_arima_fit= model_arima.fit()
            bic.append(model_arima_fit.bic)
            parameter.append(param)
        except Exception as e:
            print(e)
            continue
    d= {'Params': parameter, 'SBIC': bic }
    output = pd.DataFrame(data= d)
    result = output[output['SBIC']==output['SBIC'].min()]
    param = result['Params']
    minSBIC= result['SBIC']
    return param.values, minSBIC.values

def get_order_aic(max_range, y_train):
    '''
    Iteration to retreive the minimum AIC value to select order 
    for a model since statsmodels didn't work once I've tried
    order = get_order_aic(5, y)[0][0]
    '''
    import itertools
    aic = []
    parameter = []
    p=d=q=range(0, max_range)

    pdq = list(itertools.product(p,d,q))

    for param in pdq:
        try:
            model_arima= ARIMA(y_train, order = param) 
            model_arima_fit= model_arima.fit()
            aic.append(model_arima_fit.aic)
            parameter.append(param)
        except Exception as e:
            print(e)
            continue
    d= {'Params': parameter, 'AIC': aic }
    output = pd.DataFrame(data= d)
    result = output[output['AIC']==output['AIC'].min()]
    param = result['Params']
    minAIC= result['AIC']
    return param.values, minAIC.values


# Data storage in case we want scalate the model or keep track of values over the program run
values = {}

data = # Read your data here pd.read_excel('****.xlsx', index_col='Date').pct_change()*100 for example
data = data.dropna()
print(data.tail())
print(data.info())
print(f'Total Data: {len(data)}')
y = np.array(data['Column Selected'])

# Augmented Dickey-Fuller test for stationarity
test = adf_test(data['Column Selected'])
print("Results of Dickey-Fuller Test:")
print(adf_test(data['Column Selected']))

if (test['p-value']) <= 0.05:
    print('Series is stationary')
else:
    print('Series is not stationary. Should we differenciate it?')


# Slice your data here for train and test the model (hardcoding or develop this section) 
y_train = data[:35]
y_test = data[35:]

print('Data splitted into:\n')
print('train', len(y_train))
print('test', len(y_test))

res = arma_order_select_ic(y_train, ic=['aic', 'bic'])
order = (res.aic_min_order[0],0,res.aic_min_order[1])
print(order)


# implement ARMA
model_arima = ARIMA(y_train, order= order)
model_arima_fit = model_arima.fit()
print(model_arima_fit.summary())

# Check conditions of your model
print(smt.ArmaProcess.from_estimation(model_arima_fit).isinvertible)
print(smt.ArmaProcess.from_estimation(model_arima_fit).isstationary)

predicted = model_arima_fit.predict(0, (len(y)) , dynamic=False) # change the length or dates for your forecasting

# Essential error metrics
stats1 = np.sqrt(mean_squared_error(y, predicted[:len(y)]))
print('root mean squared error2: {}'.format(stats1) )
print('sum-of-squared residuals: {}'.format(model_arima_fit.sse) )

# Plotting predefined, customise it
plt.title(f'Your value forecast MoM ARIMA approach for *** last value predicted = {predicted[-1]:.2f}% ')
plt.plot(data[f'Column Selected'], color= 'orange',alpha=0.7, label='Sample')
plt.plot(data[f'Column Selected'],'o')
plt.plot(predicted,'--', color= 'blue', alpha=0.3, label=f'Model')
plt.plot(predicted, 'ro',)
plt.ylabel('Y label title here')
plt.axvline(y_train.index[-1], alpha =0.3 )
plt.xlabel(f'Order by Minimizing AIC {order} and RSME = {stats1:.2f}. Augmented Dickey-Fuller for unit root test p-value < 0.05.') # change if necessary
plt.xticks(rotation=45)
plt.legend()
plt.show()
plt.savefig('model.png', format='png')
plt.tight_layout()

values[f'Column Selected'] = predicted[-1] # Note the lenght for the selected value

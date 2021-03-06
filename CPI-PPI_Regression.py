import pandas_datareader as pdr
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()
sns.set_style({'axes.grid' : True, "axes.facecolor": ".9"})



def Diff(x):
    x_diff = 100*(x/x.shift(1))
    return  x_diff - 100

# Get and prepare data
data = pdr.get_data_fred('PCUOMFGOMFG', start ='01-01-2007') # Producer Price Index by Industry: Total Manufacturing Industries
data = data.rename(columns = {'PCUOMFGOMFG': 'PPI'})
data_cpi = pdr.get_data_fred('CPIAUCSL', start ='01-01-2007') # Consumer Price Index for All Urban Consumers: All Items in U.S. City Average
data['CPI'] = data_cpi
data['PPI %'] = Diff(data['PPI'])
data['CPI %'] = Diff(data['CPI'])
data = data.dropna()

X = data['PPI %']
Y = data['CPI %']



plt.figure(figsize=(10,5))
plt.plot(data.index , data['PPI'], color='b', label = 'PPI')
plt.plot(data.index, data['CPI'], color='r', label = 'CPI')
plt.title('Producer Price Index by Industry: Total Manufacturing Industries / Consumer Price Index')
plt.axhline(y=0, color='r', linestyle='-')
plt.legend()
plt.xticks(data.index, rotation=45)
plt.locator_params(axis='x', nbins=len(data)/12)
plt.ylim(data['PPI'].min() -5 , data['CPI'].max() + 5)
plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
linregression = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)
X_train = np.array(X_train).reshape(-1,1) 
y_train = np.array(y_train).reshape(-1,1) 
X_test  = np.array(X_test).reshape(-1,1) 
y_test  = np.array(y_test).reshape(-1,1) 


linregression.fit(X_train, y_train)
y_pred = linregression.predict(X_test)


alpha = linregression.intercept_
beta = linregression.coef_
r2 = linregression.score(X_train, y_train)

print(f'alpha = {alpha}')
print(f'beta = {beta}')
print(f'R^2 = {r2}')
# print(f'Real Sample = {y_test}')
# print(f'Predicted Sample = {np.round(y_pred,2)}')



plt.title('% Change on PPI & CPI')
plt.scatter(X_train, y_train) 
plt.plot(X_train, linregression.predict(X_train), 'r', label = f'y = {alpha[0]:.3f} + {beta[0][0]:.3f}x') 
plt.legend()
plt.ylabel('Return % CPI')
plt.xlabel('Return % PPI')
plt.tight_layout()
plt.show()

plt.title('Prediction Model vs. Sample')
plt.plot(y_test, label='Sample',marker='o', color = 'b')
plt.plot(y_pred, label='Predicted', marker='o', linestyle='--', color='r')
plt.xticks(list(range(len(y_pred))), rotation=45)
plt.locator_params(axis='x', nbins=len(y_pred)/3)
plt.ylabel('CPI % Change')
plt.xlabel('Steps Predicted')
plt.legend()
plt.tight_layout()
plt.show()

# Statsmodels approach
import statsmodels.formula.api as smf

formula = "Y ~ X"
model = smf.ols(formula, data = data).fit()

print(model.summary())
print(model.params)

import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
import warnings 
warnings.filterwarnings("ignore")
sns.set_theme()

# Helper functions if needed
def rmse(pred, target):
    return np.sqrt(((pred - target) ** 2).mean())

  
# Data Handling 
data = # Read your data here pd.read_excel('****.xlsx', index_col='Date').pct_change()*100 for example
data = data.dropna()
print(data.tail())
print(data.info())
y = np.array(data['Column Selected'])


# Slice your data here for train and test the model (hardcoding or develop this section) 
y_train = data[:35]
y_test = data[35:]

model = smt.ExponentialSmoothing(y_train)
res = model.fit()
pred = res.predict(0,(len(y))) # Change the lenght of your prediction here


# Plotting predefined, customise it
plt.title(f'Your value forecast MoM Exponential Smoothing approach for *** last value predicted = {pred[-1]:.2f}% ')
plt.plot(data['Column Selected'][:35], color= 'blue', alpha=0.7, label='In-the-sample Data')
plt.plot(data['Column Selected'][34:], color= 'orange', alpha=0.7, label='Out-of-the-sample Data')
plt.plot(pred, '--' ,color='green', alpha=0.5, label='Simple Exponential Smoothing')
plt.plot(data[f'Column Selected'],'o')
plt.plot(pred, 'ro')
plt.ylabel('Y label title here')
plt.axvline(y_train.index[-1], alpha =0.3 )
plt.legend()
plt.show()


# Essential error handling
stats = rmse(pred,y_train)
print('Optimal smoothing coefficient: {}'.format(res.params['smoothing_level']))
stats1 = np.sqrt(mean_squared_error(y, pred[:len(y)]))
print('root mean squared error2: {}'.format(stats1))
print('sum-of-squared residuals: {}'.format(res.sse) )

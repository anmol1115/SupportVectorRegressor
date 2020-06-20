import pandas as pd
import numpy as np
data=pd.read_csv(r'C:\\Users\Anmol\Desktop\ML Masters\Car_Purchasing_Data.csv',encoding='latin-1')

X=data.iloc[:,[3,4,5,6,7]].values
y=data.iloc[:,8:9].values

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X[:,[1,2,3,4]]=sc_X.fit_transform(X[:,[1,2,3,4]])
sc_y=StandardScaler()
y=sc_y.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=0)

from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,y)

y_pred=sc_y.inverse_transform(regressor.predict(X_test))
y_test=sc_y.inverse_transform(y_test)

import statsmodels.regression.linear_model as sm
X=np.append(arr=np.ones((500,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt.astype(float)).fit()
regressor_OLS.summary() 

X_opt=X[:,[2,3,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt.astype(float)).fit()
regressor_OLS.summary()

X_opt_train,X_opt_test,y_train,y_test=train_test_split(X_opt,y)
regressor_new=SVR(kernel='rbf')
regressor_new.fit(X_opt_train,y_train)
y_opt_pred=sc_y.inverse_transform(regressor_new.predict(X_opt_test))
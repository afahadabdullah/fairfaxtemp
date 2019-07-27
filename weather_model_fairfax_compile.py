%reset -f
import pandas as pd   
import tensorflow as tf  
from sklearn.metrics import explained_variance_score, mean_absolute_error, median_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
#df=pd.read_csv('fog.csv')
df=pd.read_csv('KAID_2010-01-01_2018-07-31_2hr.csv').set_index('date')


# s=np.nonzero(df.temperature=='-')
# s=s[0]
# df=df.drop(s)

# # df=df.drop([426])

df=df.drop(['time'],axis=1)
df = df[np.isfinite(df['wind_speed'])]
df = df[np.isfinite(df['humidity'])]
df = df[np.isfinite(df['pressure'])]
df = df[np.isfinite(df['temperature'])]

df.info()
# X will be a pandas dataframe of all columns except temperature
X = df[[col for col in df.columns if col != 'temperature']]

# y will be a pandas series of the temperature
y = df['temperature']  


# split data into training set and a temporary set using sklearn.model_selection.traing_test_split
# X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=23)

# X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=23)

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, shuffle=False)

X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, shuffle=False)


X_train.shape, X_test.shape, X_val.shape  
print("Training instances   {}, Training features   {}".format(X_train.shape[0], X_train.shape[1]))  
print("Validation instances {}, Validation features {}".format(X_val.shape[0], X_val.shape[1]))  
print("Testing instances    {}, Testing features    {}".format(X_test.shape[0], X_test.shape[1]))  

feature_cols = [tf.feature_column.numeric_column(col) for col in X.columns] 

regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,  
                                      hidden_units=[50, 50],
                                      model_dir='tf_modelx_2hr')


# def wx_input_fn(X, y=None, num_epochs=None, shuffle=True, batch_size=400):  
#     return tf.estimator.inputs.pandas_input_fn(x=X,
#                                                y=y,
#                                                num_epochs=num_epochs,
#                                                shuffle=shuffle,
#                                                batch_size=batch_size)


def wx_input_fn(X, y=None, num_epochs=None, shuffle=False, batch_size=400):  
    return tf.estimator.inputs.pandas_input_fn(x=X,
                                               y=y,
                                               num_epochs=num_epochs,
                                               shuffle=shuffle,
                                               batch_size=batch_size)


evaluations = []  
STEPS = 400  


for i in range(100):  
    regressor.train(input_fn=wx_input_fn(X_train, y=y_train), steps=STEPS)
    evaluations.append(regressor.evaluate(input_fn=wx_input_fn(X_val,y_val,num_epochs=1,shuffle=False)))

import matplotlib.pyplot as plt  


# manually set the parameters of the figure to and appropriate size
plt.rcParams['figure.figsize'] = [14, 10]

loss_values = [ev['loss'] for ev in evaluations]  
training_steps = [ev['global_step'] for ev in evaluations]

plt.scatter(x=training_steps, y=loss_values)  
plt.xlabel('Training steps (Epochs = steps / 2)')  
plt.ylabel('Loss (SSE)')  


pred = regressor.predict(input_fn=wx_input_fn(X_test,  
                                              num_epochs=1,
                                              shuffle=False))
predictions = np.array([p['predictions'][0] for p in pred])

print("The Explained Variance: %.2f" % explained_variance_score(  
                                            y_test, predictions))  
print("The Mean Absolute Error: %.2f C" % mean_absolute_error(  
                                            y_test, predictions))  
print("The Median Absolute Error: %.2f C" % median_absolute_error(  
                                            y_test, predictions))

plt.plot(range(len(y_test)), y_test)
plt.plot(range(len(y_test)), predictions)
plt.show()


d = {'dewpoint': [22], 'humidity': [58], 'wind_speed': [3], 'wind_dir_degrees': [200], 'pressure': [1012.87]}
d= pd.DataFrame(data=d)

pred = regressor.predict(input_fn=wx_input_fn(d,  
                                              num_epochs=1,
                                              shuffle=False))
predictions = np.array([p['predictions'][0] for p in pred])


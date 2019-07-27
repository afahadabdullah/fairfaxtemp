import pandas as pd  
import tensorflow as tf  
import numpy as np
d = pd.DataFrame({'dewpoint': [-11], 'humidity': [86], 'wind_speed': [24], 'wind_dir_degrees': [280], 'pressure': [1022]}, columns=['dewpoint', 'humidity', 'wind_speed','wind_dir_degrees','pressure'])


#feature_cols = [tf.feature_column.numeric_column(col) for col in X.columns] 
feature_cols = [tf.feature_column.numeric_column(col) for col in d.columns] 
regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,  
                                      hidden_units=[50, 50],
                                      model_dir='tf_modelx_2hr')


def wx_input_fn(X, y=None, num_epochs=None, shuffle=False, batch_size=400):  
    return tf.estimator.inputs.pandas_input_fn(x=X,
                                               y=y,
                                               num_epochs=num_epochs,
                                               shuffle=shuffle,
                                               batch_size=batch_size)


print('For 2hr tempreature prediction type with current data as: ')
print('model_temp(dewpoint,humidity,wind_speed,wind_dir_degrees,pressure)')

def model_temp(dew,hum,wind,dir,pres):

	d = pd.DataFrame({'dewpoint': [dew], 'humidity': [hum], 'wind_speed': [wind], 'wind_dir_degrees': [dir], 'pressure': [pres]}, columns=['dewpoint', 'humidity', 'wind_speed','wind_dir_degrees','pressure'])

	pred = regressor.predict(input_fn=wx_input_fn(d,  
	                                              num_epochs=1,
	                                              shuffle=False))
	predictions = np.array([p['predictions'][0] for p in pred])

	print('Predicted temperature (celcius) in Fairfax after 2 hr:', predictions)




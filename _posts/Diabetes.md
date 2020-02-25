<a href="https://colab.research.google.com/github/lahorekid/cnn/blob/master/Diabetes.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Diabetes Artificial Neural Network 


```python
# Importing the libraries
import keras as ks
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from keras.callbacks import ModelCheckpoint
from datetime import datetime
```


```python
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

```


```python
url = 'https://raw.githubusercontent.com/lahorekid/crypto/master/ANN-test.csv'
```


```python
dataset = pd.read_csv(url)
```


```python
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MRNO</th>
      <th>GENDER</th>
      <th>BSR</th>
      <th>AGE</th>
      <th>WEIGHT</th>
      <th>HEIGHT</th>
      <th>TEMPERATURE</th>
      <th>PULSE</th>
      <th>WAIST</th>
      <th>BMI</th>
      <th>DIAB_HIST_BURNING_FEET</th>
      <th>DIAB_HIST_DT1</th>
      <th>DIAB_HIST_DT2</th>
      <th>DIAB_HIST_NOCTURIA</th>
      <th>DIAB_HIST_POLYURIA</th>
      <th>DIAB_HIST_POLYDYPSIA</th>
      <th>DIAB_HIST_WEIGHT_LOSS</th>
      <th>DIAB_HIST_DYSPEPSIA</th>
      <th>DIAB_HIST_FEET_NUMBNESS</th>
      <th>DIAB_HIST_BLURRING_VISION</th>
      <th>DIAB_HIST_FATIGUE</th>
      <th>DIAB_HIST_IHD</th>
      <th>DAIB_RES_VESICULAR</th>
      <th>DAIB_RES_RHONCHI</th>
      <th>DAIB_RES_HARSH</th>
      <th>DAIB_RES_CREPITATION</th>
      <th>DAIB_RES_BRONCHIAL</th>
      <th>DIAG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>56</td>
      <td>MALE</td>
      <td>227</td>
      <td>67</td>
      <td>82</td>
      <td>172</td>
      <td>98.8</td>
      <td>88</td>
      <td>96</td>
      <td>27.72</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>DM1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>499</td>
      <td>FEMALE</td>
      <td>262</td>
      <td>57</td>
      <td>76</td>
      <td>152</td>
      <td>98.0</td>
      <td>86</td>
      <td>98</td>
      <td>32.89</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>DM1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>499</td>
      <td>FEMALE</td>
      <td>262</td>
      <td>57</td>
      <td>74</td>
      <td>152</td>
      <td>98.2</td>
      <td>80</td>
      <td>97</td>
      <td>32.03</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>DM1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>583</td>
      <td>MALE</td>
      <td>538</td>
      <td>48</td>
      <td>69</td>
      <td>165</td>
      <td>98.0</td>
      <td>86</td>
      <td>93</td>
      <td>25.34</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>DM2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>583</td>
      <td>MALE</td>
      <td>538</td>
      <td>48</td>
      <td>69</td>
      <td>165</td>
      <td>98.6</td>
      <td>82</td>
      <td>93</td>
      <td>25.34</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>DM2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Importing the dataset
X = dataset.iloc[:, 1:26].values
y = dataset.iloc[:, 27].values
```


```python
y
```




    array(['DM1', 'DM1', 'DM1', ..., 'DM1', 'DM1', 'DM1'], dtype=object)




```python
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])


```


```python
from sklearn.compose import ColumnTransformer


ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [0])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)

X = np.array(ct.fit_transform(X), dtype=np.float)

X = X[:, 1:]


```


```python
X
```




    array([[  1., 227.,  67., ...,   0.,   0.,   0.],
           [  0., 262.,  57., ...,   0.,   0.,   0.],
           [  0., 262.,  57., ...,   0.,   0.,   0.],
           ...,
           [  1., 360.,  43., ...,   0.,   0.,   0.],
           [  1., 360.,  43., ...,   0.,   0.,   0.],
           [  1., 360.,  43., ...,   0.,   0.,   0.]])




```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import preprocessing



labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


```


```python
y
```




    array([0, 0, 0, ..., 0, 0, 0])




```python
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_traind, y_testd = train_test_split(X, y, test_size = 0.2, random_state = 0)


y_train = ks.utils.to_categorical(y_traind, num_classes=3)
y_test = ks.utils.to_categorical(y_testd, num_classes=3)

```


```python
# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```


```python
X_test
```




    array([[-6.77366403e-01, -1.01520118e+00, -1.39871784e+00, ...,
            -6.49146046e-02, -5.20083528e-02, -6.79491222e-02],
           [ 1.47630587e+00, -1.22262070e+00,  4.95412833e-01, ...,
            -6.49146046e-02, -5.20083528e-02, -6.79491222e-02],
           [ 1.47630587e+00, -7.80726938e-01,  1.29178746e-03, ...,
            -6.49146046e-02, -5.20083528e-02, -6.79491222e-02],
           ...,
           [-6.77366403e-01,  8.33538018e-01,  1.29178746e-03, ...,
            -6.49146046e-02, -5.20083528e-02, -6.79491222e-02],
           [-6.77366403e-01, -1.16851126e+00,  4.95412833e-01, ...,
            -6.49146046e-02, -5.20083528e-02, -6.79491222e-02],
           [-6.77366403e-01,  1.03193930e+00,  7.42473355e-01, ...,
            -6.49146046e-02, -5.20083528e-02, -6.79491222e-02]])




```python
input_layer_nodes=13
hidden_layers_nodes = 350
output_layer_nodes=3
init='uniform'
optimizers='adam'
output_activation='softmax'
activation='relu'

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=input_layer_nodes, kernel_initializer=init, activation='relu', input_dim=25))

# Adding the 1 hidden layer
classifier.add(Dense(units = hidden_layers_nodes, kernel_initializer = init, activation='relu'))

# Adding the 2 hidden layer
classifier.add(Dense(units = hidden_layers_nodes, kernel_initializer = init, activation='relu'))

# Adding the 3 hidden layer
classifier.add(Dense(units = hidden_layers_nodes, kernel_initializer = init, activation='relu'))

# Adding the 4 hidden layer
classifier.add(Dense(units = hidden_layers_nodes, kernel_initializer = init, activation='relu'))

# Adding the 5 hidden layer
classifier.add(Dense(units = hidden_layers_nodes, kernel_initializer = init, activation='relu'))

# Adding the 6 hidden layer
classifier.add(Dense(units = hidden_layers_nodes, kernel_initializer = init, activation='relu'))

# Adding the 7 hidden layer
classifier.add(Dense(units = hidden_layers_nodes, kernel_initializer = init, activation='relu'))

# Adding the 8 hidden layer
classifier.add(Dense(units = hidden_layers_nodes, kernel_initializer = init, activation='relu'))

# Adding the 9 hidden layer
classifier.add(Dense(units = hidden_layers_nodes, kernel_initializer = init, activation='relu'))

# Adding the 10 hidden layer
classifier.add(Dense(units = hidden_layers_nodes, kernel_initializer = init, activation='relu'))


# Adding the output layer
classifier.add(Dense(units = output_layer_nodes, kernel_initializer = init, activation=output_activation))

# Compiling the ANN
classifier.compile(optimizer=optimizers, loss='categorical_crossentropy', metrics=['accuracy'])
```


```python
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, validation_split=0.15, batch_size=20, epochs=2, callbacks=callbacks_list)
#classifier.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=10, epochs=250, callbacks=callbacks_list)
```

    Train on 8507 samples, validate on 1502 samples
    Epoch 1/2
    8507/8507 [==============================] - 12s 1ms/step - loss: 0.7665 - acc: 0.7116 - val_loss: 0.7167 - val_acc: 0.7264
    
    Epoch 00001: val_acc did not improve from 0.74834
    Epoch 2/2
    8507/8507 [==============================] - 11s 1ms/step - loss: 0.7129 - acc: 0.7323 - val_loss: 0.7152 - val_acc: 0.7344
    
    Epoch 00002: val_acc did not improve from 0.74834





    <keras.callbacks.History at 0x7f29e9e14780>




```python
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
```


```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.75      0.99      0.85      1749
               1       1.00      0.00      0.01       505
               2       0.00      0.00      0.00       249
    
       micro avg       0.75      0.69      0.72      2503
       macro avg       0.58      0.33      0.29      2503
    weighted avg       0.73      0.69      0.60      2503
     samples avg       0.69      0.69      0.69      2503
    


    /usr/local/lib/python3.6/dist-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /usr/local/lib/python3.6/dist-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in samples with no predicted labels. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))


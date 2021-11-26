import numpy
import pandas as pd 
from itertools import chain
import pickle

path = 'covid-symptoms-dataset.csv'
df = pd.read_csv(path)

pd.DataFrame(df.dtypes).rename(columns = {0:'dtype'})
df_OHE = pd.get_dummies(df, drop_first=True) 
df_OHE = df_OHE.drop(['Running Nose_Yes', 'Asthma_Yes', 'Chronic Lung Disease_Yes',
                      'Abroad travel_Yes', 'Headache_Yes', 'Heart Disease_Yes', 
                      'Attended Large Gathering_Yes', 'Diabetes_Yes', 'Fatigue _Yes', 
                      'Contact with COVID Patient_Yes', 'Gastrointestinal _Yes', 'Visited Public Exposed Places_Yes', 
                      'Visited Public Exposed Places_Yes', 
                      'Family working in Public Exposed Places_Yes'], axis=1)

df_OHE.rename(columns = {'Dry Cough_Yes':'Dry Cough'}, inplace=True)
df_OHE.rename(columns = {'Fever_Yes':'Fever'}, inplace=True)
df_OHE.rename(columns = {'Breathing Problems_Yes':'Breathing Problem'}, inplace=True)
df_OHE.rename(columns = {'Sore Throat_Yes':'Sore Throat'}, inplace=True)
df_OHE.rename(columns = {'Hyper Tension_Yes':'Hyper Tension'}, inplace=True)
df_OHE.rename(columns = {'COVID-19_Yes':'COVID-19'}, inplace=True)


X = df_OHE.drop('COVID-19', axis=1)
y = df_OHE['COVID-19']

def prepare_init_features(features_ls):
    X_user = numpy.zeros(5)
    features_dict = {
        'Dry Cough_Yes': 0, 
        'Fever_Yes': 1, 
        'Breathing Problems_Yes': 2,
        'Sore Throat_Yes': 3,
        'Hyper Tension_Yes': 4
    }
    columns = numpy.array(['Dry Cough_Yes', 'Fever_Yes', 'Breathing Problems_Yes', 'Sore Throat_Yes', 'Hyper Tension_Yes'])
    for feature in features_ls:
        if feature in columns:
            X_user[int(features_dict[feature])] = 1
        else:
            X_user[int(features_dict[feature])] = 0
    
    return [X_user]

class MinMaxScaler(object):
    def __init__(self, feature_range=(0, 1)):
        self._low, self._high = feature_range

    def fit(self, X):
        self._min = X.min(axis=0)
        self._max = X.max(axis=0)
        return self

    def transform(self, X):
        X_std = (X - self._min) / (self._max - self._min)
        return X_std * (self._high - self._low) + self._low

    def fit_transform(self, X):
        return self.fit(X).transform(X)

def _indexing(x, indices):
    # np array indexing
    if hasattr(x, 'shape'):
        return x[indices]

    # list indexing
    return [x[idx] for idx in indices]

def train_test_split(*arrays, test_size=0.25, shufffle=True, random_seed=1):
    import numpy as np 

    assert 0 < test_size < 1
    assert len(arrays) > 0
    
    length = len(arrays[0])
    for i in arrays:
        assert len(i) == length

    n_test = int(np.ceil(length*test_size))
    n_train = length - n_test

    if shufffle:
        perm = np.random.RandomState(random_seed).permutation(length)
        test_indices = perm[:n_test]
        train_indices = perm[n_test:]
    else:
        train_indices = np.arange(n_train)
        test_indices = np.arange(n_train, length)

    return list(chain.from_iterable((_indexing(x, train_indices), _indexing(x, test_indices)) for x in arrays))

class LogisticRegression:
        
    def sigmoid(self, z): 
      return 1 / (1 + numpy.e**(-z))
    
    def cost_function(self, X, y, weights):                 
        z = numpy.dot(X, weights)
        predict_1 = y * numpy.log(self.sigmoid(z))
        predict_0 = (1 - y) * numpy.log(1 - self.sigmoid(z))
        return -sum(predict_1 + predict_0) / len(X)
    
    def fit(self, X, y, epochs=25, lr=0.05):        
        loss = []
        weights = numpy.random.rand(X.shape[1])
        N = len(X)
                 
        for _ in range(epochs):        
            # Gradient Descent
            y_hat = self.sigmoid(numpy.dot(X, weights))
            weights -= lr * numpy.dot(X.T,  y_hat - y) / N            
            # Saving Progress
            loss.append(self.cost_function(X, y, weights)) 
            
        self.weights = weights
        self.loss = loss
    
    def predict(self, X):        
        # Predicting with sigmoid function
        z = numpy.dot(X, self.weights)
        # Returning binary result
        return [1 if i > 0.5 else 0 for i in self.sigmoid(z)]

scaler = MinMaxScaler(feature_range=(-1, 1))
X_array = X.to_numpy()
X_scaled = scaler.fit_transform(X_array)

logreg = LogisticRegression()
logreg.fit(X_scaled, y, epochs=500, lr=0.5)

pickle.dump(logreg, open('model_logreg.pkl','wb'))

model = pickle.load(open('model_logreg.pkl','rb'))
print(model.predict([[1., 1., 1., -1., 1.]]))




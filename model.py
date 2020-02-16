import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('Housing.csv')
print(df.info())
print(df.columns)

df['Avg. Area'].fillna(df['Avg. Area'].mean(), inplace=True)
df['No. of BedRooms'].fillna(0, inplace=True)

#Converting words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

df['No. of BedRooms'] = df['No. of BedRooms'].apply(lambda x: convert_to_int(x))

X = df.iloc[:,0:3]
y = df.iloc[:,-1]

from sklearn.linear_model import LinearRegression
model = LinearRegression()

#Fitting model with trainig data
model.fit(X, y)

# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[800, 9, 6]]))

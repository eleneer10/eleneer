import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('C:/Users/Ijaz10/Desktop/CSV/drug200.csv')

y = df[['Drug']]
X = df.drop('Drug', axis = 1)

encoder = OneHotEncoder(handle_unknown = 'ignore')
y_new = encoder.fit_transform(y)
y_new_array = y_new.toarray()       #encoded output attributes

num_attr = df[['Age', 'Na_to_K']].columns 
cat_attr = ['Sex', 'BP', 'Cholesterol']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'median')),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('c_imp', SimpleImputer(strategy = 'most_frequent')),
    ('c_enc', OneHotEncoder(handle_unknown = 'ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_attr),
    ('cat', cat_pipeline, cat_attr)
])

clf = DecisionTreeClassifier(criterion = 'gini', max_depth = 2)
model = Pipeline([
    ('prep', preprocessor),
    ('clf', clf)
])

model.fit(X, y_new_array)

#Test Data input

Age = int(input('Enter the Age : '))
Sex = str(input('Enter the gender : '))
BP = str(input('Enter BP level : '))
cl = str(input('Enter Cholesterol level : '))
na_k = float(input('Enter na_k level : '))

test = pd.DataFrame([[Age, Sex, BP, cl, na_k]], columns = X.columns)
y_pred = model.predict(test)

y_out = encoder.inverse_transform(y_pred)

print('Recommended Drug : ', ''.join(map(str, y_out.ravel())))

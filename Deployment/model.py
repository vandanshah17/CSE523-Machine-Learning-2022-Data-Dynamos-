import pandas as pd
import json
import os
import re
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report


def load_data(fp):
  with open(fp) as fid:
    series = (pd.Series(json.loads(s)) for s in fid)
    return pd.concat(series,axis=1).T

renththerunway_fp = "D:/ML Project/renttherunway_final_data.json"
df = load_data(renththerunway_fp)

#--------------------------------------------------------------------

target_feature = 'fit'
n_classes = df[target_feature].unique().shape[0]
df.head()

#--------------------------------------------------------------------

def check_nulls(data):
    for col in df:
        print(f'Column \'{col}\'. Is null - {data[col].isnull().sum()}')
      
#--------------------------------------------------------------------

to_drop = df[df['fit'] == 'fit'].isnull().any(axis=1)
n = to_drop.sum()
to_drop.shape, df.shape
df = df.drop(df[df['fit'] == 'fit'][to_drop].index, axis=0)
check_nulls(df)

#--------------------------------------------------------------------

def parse_ht(height):
    ht_ = height.split("' ")
    ft_ = float(ht_[0])
    in_ = float(ht_[1].replace("\"",""))
    return (12*ft_) + in_

def pounds_to_kilos(s):
    return int(s.replace('lbs', '')) * 0.45359237

df['height'] = (df['height']
                        .fillna("0' 0\"")
                        .apply(parse_ht))
df['height'][df['height'] == 0] = df['height'].median()

df['weight'] = (df['weight']
                        .fillna('0lbs')
                        .apply(pounds_to_kilos))
df['weight'][df['weight'] == 0.0] = df['weight'].median()

df['user_id'] = pd.to_numeric(df['user_id'])
df['bust size'] = df['bust size'].fillna(df['bust size'].value_counts().index[0])
df['body type'] = df['body type'].fillna(df['body type'].value_counts().index[0])
df['item_id'] = pd.to_numeric(df['item_id'])
df['size'] = pd.to_numeric(df['size'])

df['age'] = pd.to_numeric(df['age'])
df['age'] = df['age'].fillna(df['age'].median())

df['rating'] = pd.to_numeric(df['rating'])
df['rating'] = df['rating'].fillna(df['rating'].median())

df['review_date'] = pd.to_datetime(df['review_date'], format='%B %d, %Y')

#--------------------------------------------------------------------

#column mapper
col_mapper = {
    'bust size': 'bust_size',
    'weight': 'usr_weight_kg',
    'rating': 'review_rating',
    'rented for': 'rented_for',
    'body type': 'body_type',
    'category': 'product_category',
    'height': 'usr_height_inchs',
    'size': 'product_size',
    'age': 'usr_age',
}
df.rename(col_mapper, axis=1, inplace=True)

#--------------------------------------------------------------------

newdf = df.copy()

#--------------------------------------------------------------------

#bust size and category mapper
def parse_bust_size(s):
    m = re.match(r'(\d+)([A-Za-z])(\+?)', s)
    if m:
        return pd.Series(data=[int(m.group(1)), m.group(2).lower()])
    return []

mapper = {
    0: 'bust_size_num', 
    1: 'bust_size_cat'
}

temp_df = newdf['bust_size'].apply(parse_bust_size).rename(mapper, axis=1)
temp_df['bust_size_num'] = pd.to_numeric(temp_df['bust_size_num'])
newdf = newdf.join(temp_df)
newdf.drop(['bust_size'], axis=1, inplace=True)

#--------------------------------------------------------------------

#bust category mapper
mapper = {
    'a': 1,
    'b': 2,
    'c': 3,
    'd': 4,
    'e': 5,
    'f': 6,
    'g': 7,
    'h': 8,
    'i': 9,
    'j': 10,
}
newdf['bust_size_cat'] = newdf['bust_size_cat'].map(mapper)
newdf.head()
#--------------------------------------------------------------------

mapper = {
    'small': -1,
    'fit': 0,
    'large': 1,
}
newdf['fit'] = newdf['fit'].map(mapper)

#--------------------------------------------------------------------

numeric_dtypes = {'int64', 'float64'}
numeric_features = [c for c in newdf.columns if str(newdf[c].dtype) in numeric_dtypes]
numeric_features.remove('user_id')
numeric_features.remove('item_id')
numeric_features.remove('review_rating')

#--------------------------------------------------------------------

newdf['BMI'] = newdf['usr_weight_kg'] / np.power(newdf['usr_height_inchs'], 2)
newdf.drop(['usr_weight_kg', 'usr_height_inchs'], axis=1, inplace=True)

#--------------------------------------------------------------------

numeric_features.append('BMI')
numeric_features.remove('usr_weight_kg')
numeric_features.remove('usr_height_inchs')

#--------------------------------------------------------------------

rented_for_col = newdf['rented_for'].replace('party: cocktail', 'other')
rented_for_col_encoded = pd.get_dummies(rented_for_col)
rented_for_col_encoded.head()

#--------------------------------------------------------------------

body_type_col = newdf['body_type']
body_type_col_encoded = pd.get_dummies(body_type_col)
body_type_col_encoded.head()

#--------------------------------------------------------------------

counts = newdf['product_category'].value_counts()

#--------------------------------------------------------------------

newdf['product_category'].nunique()
threshold = 1000

#--------------------------------------------------------------------

repl = counts[counts <= threshold].index

#--------------------------------------------------------------------

top_n = 8
prod_cats = newdf['product_category'].value_counts().index[:top_n].values
prod_cat_col = newdf['product_category'].apply(lambda x: x if x in prod_cats else 'other')
prod_cat_col_encoded = pd.get_dummies(newdf['product_category'].replace(repl, 'uncommon'))

#--------------------------------------------------------------------

dummy_features_df = pd.concat((prod_cat_col_encoded, rented_for_col_encoded, body_type_col_encoded), axis=1)
dummy_features_df.head()

#--------------------------------------------------------------------

# removing all features having in 80% samples either ones or zeros
sel = VarianceThreshold(threshold=(0.8 * (1 - 0.8)))
dummy_features = sel.fit_transform(dummy_features_df)

#--------------------------------------------------------------------

data = pd.concat((newdf[numeric_features], dummy_features_df), axis=1)

head, tail = os.path.split(renththerunway_fp)
cleaned_data_fp = 'cleaned_' + tail.replace('json', 'csv')

data.to_csv(cleaned_data_fp, index=False)
data.head()


#--------------------------------------------------------------------

X = np.hstack((newdf[numeric_features].drop([target_feature,'bust_size_num', 'bust_size_cat'], axis=1), dummy_features))
y = newdf["fit"].astype('float32').values

#--------------------------------------------------------------------

smote_nc = SMOTENC(categorical_features=[0, 2], random_state=0)
X_resampled, y_resampled = smote_nc.fit_resample(X, y)

#--------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

#--------------------------------------------------------------------

#rfc = RandomForestClassifier(n_estimators=50, random_state=2)
#rfc.fit(X_train, y_train)

# Make predictions on the test set
#y_pred = rfc.predict(X_test)

# Calculate the accuracy of the model
#accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy: ", accuracy)
#--------------------------------------------------------------------

#print(classification_report(y_test, y_pred))
#X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
print(X_resampled.shape)
print(X_train.shape)

# Define the parameter grid for hyperparameter tuning
#param_grid = {'n_estimators': [50, 100, 200],'max_features': ['sqrt', 'log2'],'max_depth': [5, 10, 20, 30],'min_samples_split': [2, 5, 10],'min_samples_leaf': [1, 2, 4],'bootstrap': [True, False]}

# Create a random forest classifier
rfc = RandomForestClassifier(n_estimators=50,random_state=2)

# Create a grid search object  
#grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1)

# Fit the grid search object to the data
#grid_search.fit(X_train, y_train)
rfc.fit(X_train,y_train)

# Print the best hyperparameters and their corresponding accuracy score
#print("Best hyperpara: ", grid_search.best_params_)
#print("Accuracy Score: ", grid_search.best_params_)


#grid_results = grid_search.fit(X_train, y_train)
#final_model = rfc.set_param (**grid_results.best_params_)
#final_model.fit(X_train, y_train)
y_pred = rfc.predict(X_test) 

#--------------------------------------------------------------------

#test_scores = []
#train_scores = []
#for i in range(2,20):
 #   knn = KNeighborsClassifier(i)
  #  knn.fit(X_train,y_train)    
   # train_scores.append(knn.score(X_train,y_train))
    #test_scores.append(knn.score(X_test,y_test))
#max_test_score = max(test_scores)
#test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
#print('Max test score {} % and k = {}'.format(max_test_score*100, list(map(lambda x: x+1, test_scores_ind))))

#--------------------------------------------------------------------

pickle.dump(rfc, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl','rb'))
# print(model.predict([[2, 9, 6]]))
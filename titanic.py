#https://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Binarizer

from sklearn.preprocessing import StandardScaler
import pandas as pd

X = pd.read_csv("titanic_train.csv")
y = X.pop("Survived")

test = pd.read_csv("titanic_test.csv")


#print(test.describe())

X["Age"].fillna(X["Age"].mean(), inplace=True)
#print(X.describe())

test["Fare"].fillna(test["Fare"].mean(), inplace=True)
test.describe()

numeric_variables=list(X.dtypes[X.dtypes!="object"].index)
#print(X[numeric_variables].head())

#Standardize features by removing the mean and scaling to unit variance
scaled_X = StandardScaler().fit(X[numeric_variables]).transform(X[numeric_variables])
#print(scaled_X)

model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
#model.fit(X[numeric_variables], y)
model.fit(scaled_X, y)

#print(model.oob_score_)


y_oob = model.oob_prediction_
#print("c_stat: ", roc_auc_score(y, y_oob))


def describe_categorical(X):
    from IPython.display import display, HTML
    display(HTML(X[X.columns[X.dtypes == "object"]].describe().to_html()))
    
describe_categorical(X)

W=describe_categorical(test)

X.drop(["Name", "Ticket", "PassengerId"], axis=1, inplace=True)
test.drop(["Name", "Ticket"], axis=1, inplace=True)
id = test.pop("PassengerId")    
    
def clean_cabin(x):
    try:
        return x[0]
    except TypeError:
        return "None"
    
X["Cabin"] = X.Cabin.apply(clean_cabin)
test["Cabin"] = test.Cabin.apply(clean_cabin)

X.Cabin
test.Cabin 
    
categorical_variables = ["Sex", "Cabin", "Embarked"]

for variable in categorical_variables:
    X[variable].fillna("Missing", inplace=True)
    dummies = pd.get_dummies(X[variable], prefix=variable)
    X = pd.concat([X, dummies], axis=1)
    X.drop([variable], axis=1, inplace=True)
    
#print(X)        
    
def printall(X, max_rows=10):
    from IPython.display import display, HTML
    display(HTML(X.to_html(max_rows=max_rows)))
    
#printall(X)    
    
for variable in categorical_variables:
    test[variable].fillna("Missing", inplace=True)
    dummies = pd.get_dummies(test[variable], prefix=variable)
    test = pd.concat([test, dummies], axis=1)
    test.drop([variable], axis=1, inplace=True)
    
all_cols = X.columns.union(test.columns)

X = X.assign(**{col:0 for col in all_cols.difference(X.columns).tolist()})
test = test.assign(**{col:0 for col in all_cols.difference(test.columns).tolist()})

scaled_X = StandardScaler().fit(X).transform(X)

model = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=-1, random_state=42)
#model.fit(X, y)
model.fit(scaled_X, y)
print("c_stat: ", roc_auc_score(y, model.oob_prediction_))

model.feature_importances_

feature_importances = pd.Series(model.feature_importances_, index=X.columns)
print(feature_importances)
feature_importances.sort_values(inplace=True)
feature_importances.plot(kind='barh', figsize=(7,6))

def graph_feature_importances(model, feature_names, autoscale=True, headroom=0.05, width=10, summarized_columns=None):
    if autoscale:
        x_scale = model.feature_importances_.max() + headroom
    else:
        x_scale = 1
        
    feature_dict=dict(zip(feature_names, model.feature_importances_))
    
    if summarized_columns:
        for col_name in summarized_columns:
            sum_value = sum(x for i, x in feature_dict.items() if col_name in i )
            keys_to_remove = [i for i in feature_dict.keys() if col_name in i ]
            for i in keys_to_remove:
                feature_dict.pop(i)
            feature_dict[col_name] = sum_value
    results = pd.Series(feature_dict, index=feature_dict.keys())
    results.sort_values(inplace=True)
    print(results)
    results.plot(kind='barh', figsize=(width, len(results)/4), xlim=(0, x_scale))
    
graph_feature_importances(model, X.columns, summarized_columns=categorical_variables)


model = RandomForestRegressor(1000, oob_score=True, n_jobs=1, random_state=42)
#model.fit(X, y)
model.fit(scaled_X, y)

results = []
n_estimator_options = [30, 50, 100, 200, 500, 1000, 2000]

for trees in n_estimator_options:
    model = RandomForestRegressor(trees, oob_score=True, n_jobs=-1, random_state=42)
    #model.fit(X, y)
    model.fit(scaled_X, y)
    print(trees, 'trees')
    roc = roc_auc_score(y, model.oob_prediction_)
    print('C-stat: ', roc)
    results.append(roc)
    print (" ")
    
pd.Series(results, n_estimator_options).plot()


results = []
max_features_options = ["auto", None, "sqrt", "log2", 0.9, 0.2]

for max_features in max_features_options:
    model = RandomForestRegressor(n_estimators=1000, oob_score=True, n_jobs=-1, random_state=42, max_features=max_features)
    #model.fit(X, y)
    model.fit(scaled_X, y)
    print(max_features, "option")
    roc = roc_auc_score(y, model.oob_prediction_)
    print('C-stat: ', roc)
    results.append(roc)
    print (" ")
    
pd.Series(results, max_features_options).plot(kind='barh', xlim=(.85, .88))


results = []
min_samples_leaf_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for min_samples in min_samples_leaf_options:
    model = RandomForestRegressor(n_estimators=1000, oob_score=True, n_jobs=-1, random_state=42, max_features="auto", min_samples_leaf=min_samples)
    #model.fit(X, y)
    model.fit(scaled_X, y)
    print(min_samples, "min samples")
    roc = roc_auc_score(y, model.oob_prediction_)
    print('C-stat: ', roc)
    results.append(roc)
    print (" ")
    
pd.Series(results, min_samples_leaf_options).plot()


model = RandomForestRegressor(n_estimators=1000, 
                              oob_score=True, 
                              n_jobs=-1, 
                              random_state=42, 
                              max_features="auto", 
                              min_samples_leaf=5)
#model.fit(X, y)
model.fit(scaled_X, y)
roc = roc_auc_score(y, model.oob_prediction_)
print('C-stat: ', roc)

test_X = scaled_X[:10]
pred = model.predict(test_X)
print(pred)
binary_pred = Binarizer(threshold=0.55).fit(pred).transform(pred)
print(binary_pred)




    
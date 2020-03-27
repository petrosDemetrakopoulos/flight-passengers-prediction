#Πέτρος - Σώζων Δημητρακόπουλος - Α.Μ 3150034 
#Σταύρος Μαρκόπουλος - Α.Μ 3150098 
#Νίκος Καβαλέγας - Α.Μ 6130034 (Στατιστική)

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import metrics
from geopy import distance
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.utils.class_weight import compute_class_weight
from xgboost.sklearn import XGBClassifier

#Import train and test Data
df_train = pd.read_csv("dataset/train.csv")
df_test = pd.read_csv("dataset/test.csv")

#Edit train Data
df_train['dt'] = df_train['DateOfDeparture'].apply(lambda x: datetime.strptime(x,"%Y-%m-%d"))
df_train['day'] = df_train['dt'].map(lambda x: x.day)
df_train['month'] = df_train['dt'].map(lambda x: x.month)
df_train['year'] = df_train['dt'].map(lambda x: x.year)
df_train['week'] = df_train['dt'].apply(lambda x: x.isocalendar()[1])
df_train['weekday'] = df_train['dt'].apply(lambda x: x.weekday())
df_train['weekend'] = df_train['weekday'].map(lambda x: x == 5)
df_train['nye'] = df_train['year'].map(lambda x: datetime(x,12,31))
df_train['indipendence'] = df_train['year'].map(lambda x: datetime(x,7,4))
df_train['dateDistance'] = df_train['dt'].apply(lambda date: (date - pd.to_datetime("2010-01-01")).days)
for i in range(df_train.shape[0]):
	start = (df_train.loc[i,'LongitudeDeparture'], df_train.loc[i,'LatitudeDeparture'])
	stop = (df_train.loc[i,'LongitudeArrival'],df_train.loc[i,'LatitudeArrival'])
	df_train.loc[i,'distance'] = distance.distance(start, stop).km
for i in range(df_train.shape[0]):
    start = df_train.loc[i,'nye']
    stop = df_train.loc[i,'dt']
    delta = start - stop
    df_train.loc[i,'days_to_nye'] = abs(int(delta.days))
for i in range(df_train.shape[0]):
    start = df_train.loc[i,'indipendence']
    stop = df_train.loc[i,'dt']
    delta = start - stop
    df_train.loc[i,'days_to_ind'] = abs(int(delta.days))
for i in range(df_train.shape[0]):
    start = df_train.loc[i,'CityDeparture']
    stop = df_train.loc[i,'CityArrival']
    df_train.loc[i,'route'] = start+stop
df_train['times'] = df_train.groupby('route')['route'].transform('count')

#Edit test Data
df_test['dt'] = df_test['DateOfDeparture'].apply(lambda x: datetime.strptime(x,"%Y-%m-%d"))
df_test['day'] = df_test['dt'].map(lambda x: x.day)
df_test['month'] = df_test['dt'].map(lambda x: x.month)
df_test['year'] = df_test['dt'].map(lambda x: x.year)
df_test['week'] = df_test['dt'].apply(lambda x: x.isocalendar()[1])
df_test['weekday'] = df_test['dt'].apply(lambda x: x.weekday())
df_test['weekend'] = df_test['weekday'].map(lambda x: x == 5)
df_test['nye'] = df_test['year'].map(lambda x: datetime(x,12,31))
df_test['indipendence'] = df_test['year'].map(lambda x: datetime(x,7,4))
df_test['dateDistance'] = df_test['dt'].apply(lambda date: (date - pd.to_datetime("2010-01-01")).days)
for i in range(df_test.shape[0]):
	start = (df_test.loc[i,'LongitudeDeparture'], df_test.loc[i,'LatitudeDeparture'])
	stop = (df_test.loc[i,'LongitudeArrival'],df_test.loc[i,'LatitudeArrival'])
	df_test.loc[i,'distance'] = distance.distance(start, stop).km
for i in range(df_test.shape[0]):
    start = df_test.loc[i,'nye']
    stop = df_test.loc[i,'dt']
    delta = start - stop
    df_test.loc[i,'days_to_nye'] = abs(int(delta.days))
for i in range(df_test.shape[0]):
    start = df_test.loc[i,'indipendence']
    stop = df_test.loc[i,'dt']
    delta = start - stop
    df_test.loc[i,'days_to_ind'] = abs(int(delta.days))
for i in range(df_test.shape[0]):
    start = df_test.loc[i,'CityDeparture']
    stop = df_test.loc[i,'CityArrival']
    df_test.loc[i,'route'] = start+stop
df_test['times'] = df_test.groupby('route')['route'].transform('count')

#Transform string data to numeric
le = LabelEncoder()
le.fit(df_test['CityDeparture'])
df_test['CityDeparture'] = le.transform(df_test['CityDeparture'].astype(str))
le.fit(df_train['CityDeparture'])
df_train['CityDeparture'] = le.transform(df_train['CityDeparture'].astype(str))
le.fit(df_test['CityArrival'])
df_test['CityArrival'] = le.transform(df_test['CityArrival'].astype(str))
le.fit(df_train['CityArrival'])
df_train['CityArrival'] = le.transform(df_train['CityArrival'].astype(str))
le = LabelEncoder()
le.fit(df_test['route'])
df_test['route'] = le.transform(df_test['route'].astype(str))
le.fit(df_train['route'])
df_train['route'] = le.transform(df_train['route'].astype(str))
le = LabelEncoder()
le.fit(df_test['weekend'])
df_test['weekend'] = le.transform(df_test['weekend'].astype(bool))
le.fit(df_train['weekend'])
df_train['weekend'] = le.transform(df_train['weekend'].astype(bool))

#Use One-Hot encodding
df_train = df_train.join(pd.get_dummies(df_train['Departure'], prefix='dep'))
df_train = df_train.join(pd.get_dummies(df_train['Arrival'], prefix='arr'))
df_train = df_train.drop('Departure', axis=1)
df_train = df_train.drop('Arrival', axis=1)
df_test = df_test.join(pd.get_dummies(df_test['Departure'], prefix='dep'))
df_test = df_test.join(pd.get_dummies(df_test['Arrival'], prefix='arr'))
df_test = df_test.drop('Departure', axis=1)
df_test = df_test.drop('Arrival', axis=1)
df_train = df_train.join(pd.get_dummies(df_train['CityDeparture'], prefix='cityDep'))
df_train = df_train.join(pd.get_dummies(df_train['CityArrival'], prefix='cityArr'))
df_train = df_train.drop('CityDeparture', axis=1)
df_train = df_train.drop('CityArrival', axis=1)
df_test = df_test.join(pd.get_dummies(df_test['CityDeparture'], prefix='cityDep'))
df_test = df_test.join(pd.get_dummies(df_test['CityArrival'], prefix='cityArr'))
df_test = df_test.drop('CityDeparture', axis=1)
df_test = df_test.drop('CityArrival', axis=1)
df_train = df_train.join(pd.get_dummies(df_train['year'], prefix='year'))
df_train = df_train.join(pd.get_dummies(df_train['month'], prefix='month'))
df_train = df_train.join(pd.get_dummies(df_train['weekday'], prefix='weekday'))
df_train = df_train.join(pd.get_dummies(df_train['week'], prefix='week'))
df_test = df_test.join(pd.get_dummies(df_test['year'], prefix='year'))
df_test = df_test.join(pd.get_dummies(df_test['month'], prefix='month'))
df_test = df_test.join(pd.get_dummies(df_test['weekday'], prefix='weekday'))
df_test = df_test.join(pd.get_dummies(df_test['week'], prefix='week'))
df_train = df_train.join(pd.get_dummies(df_train['route'], prefix='route'))
df_train = df_train.drop('route', axis=1)
df_test = df_test.join(pd.get_dummies(df_test['route'], prefix='route'))
df_test = df_test.drop('route', axis=1)

#Select columns
features = df_train.drop(['DateOfDeparture','weekend','times','PAX','days_to_ind','indipendence','dt','LatitudeDeparture','LatitudeArrival','LongitudeArrival','LongitudeDeparture','std_wtd','nye', 'indipendence','day','WeeksToDeparture','year'], axis=1)
features_test = df_test.drop(['DateOfDeparture','weekend','times','days_to_ind','indipendence','dt','LatitudeDeparture','LatitudeArrival','LongitudeArrival','LongitudeDeparture','std_wtd','nye', 'indipendence','day','WeeksToDeparture','year'], axis=1)
X_test = features_test[features.columns]
X_train = features.values
Y_train = df_train[['PAX']]
X_train, X_dev, y_train, y_dev = train_test_split(X_train, Y_train, test_size=0.01)

#Create models
cw = compute_class_weight(class_weight='balanced',classes=np.asarray([0,1,2,3,4,5,6,7]), y =y_train.values.ravel())
cw_dict = {}
for i in range(0,8):
    cw_dict[i] = cw[i] #calculating class weights
clf = tree.DecisionTreeClassifier(max_features= 0.2, min_samples_leaf= 1, min_samples_split= 2, random_state= 2018)
bagging = BaggingClassifier(clf, max_samples=1.0)
rf = RandomForestClassifier(min_samples_split=2, min_samples_leaf=1,bootstrap=True, max_features='auto', max_depth=20, class_weight = cw_dict, n_estimators=1000,verbose=1, n_jobs=-1)
grb = GradientBoostingClassifier(learning_rate = 0.1, n_estimators=100, max_features=0.6, verbose=1)
xgb = XGBClassifier(num_class = 8, estimators = 400, max_depth=20, learning_rate=0.01, verbose = 1, n_jobs=-1)
ab = AdaBoostClassifier(RandomForestClassifier(n_estimators = 200,class_weight = cw_dict, max_features='auto', verbose = 1,min_samples_split=2,max_depth = 20, n_jobs = -1),
                         algorithm="SAMME",
                         n_estimators=100)

#ensemble algorithms
ensemble = VotingClassifier(estimators=[('rf', rf), ('grb',grb), ('bagging', bagging), ('ab',ab), ('xgb',xgb)], voting='soft')
ensemble.fit(X_train,y_train.values.ravel())

#predict local test
y_pred = ensemble.predict(X_dev)
print("Accuracy:",metrics.accuracy_score(y_dev, y_pred))
print("f1-score:",metrics.f1_score(y_dev, y_pred, average='micro'))
'''
#predict kaggle test
y_pred_test = ensemble.predict(X_test.values)
resp = pd.DataFrame(data={'Label':y_pred_test})
resp.to_csv("resp_f1.csv", sep=',', encoding='utf-8')
print("Accuracy:",metrics.accuracy_score(y_dev, y_pred))
print("f1-score:",metrics.f1_score(y_dev, y_pred, average='micro'))
'''

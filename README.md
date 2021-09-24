# INF131: The case of flight passengers prediction
A machine learning  model (classifier) that predicts number of passengers in flights.

# Team members

Petros Demetrakopoulos ,
Stavros Markopoulos,
Nikos Kavalegas,

# Project description 
A supervised learning problem given as a project in the "Data Mining in Dtabases and World Wide Web" course in Computer Science Department of AUEB. It is a classification problem that asks to determine the 'PAX' variable (that can get 8 discrete values) for a set of flights that happened in U.S.A. Each flight contains some features (Arrival / Departure airport, date etc.) and for some flights 'PAX' variable is already determined (so we use them as our training dataset). 
We have to keep in mind that 'PAX' (passengers approximately) variable is strongly related to the number of passengers a flight is carrying. The goal of the project is to predict the value of 'PAX' variable for the flights that it is not already predicted.

# Dataset
```train.csv``` file contains the tranining dataset and ```test.csv``` contains the unlabeled test set.
Each line contains a 'flight' that has the following features:

* DateOfDeparture: Date of the flight departure
* Departure: Airport code of the departure airport
* CityDeparture: Name of departure airport
* LongitudeDeparture: Longitude of the departure airport
* LatitudeDeparture: Latitude of the departure airport
* Arrival: Airport code of the arrival airport
* CityArrival: Name of arrival airport
* LongitudeArrival: Longitude of the arrival airport
* LatitudeArrival: Latitude of the arrival airport
* WeeksToDeparture: The average number of weeks before the passengers booked their tickets for the specific flight
* std_wtd: Standard deviation for the 'WeeksToDeparture' variable

Training set also contains 'PAX' variable as we mentioned above which is strongly related to the number of passengers a flight is carrying. This variable can have 8 discrete values (0 to 7).

`PAX` is the variable we need to predict in the test set.

# Feature extraction
From the features that the dataset originally contained, we extracted the following derivative features that helped out model achieve much better results:

* year
* month
* weekday
* week, the number of the week the flight is happening
* weekend, a flag indicating if the flight is happening during a weekend
* route, a variable indicating which 'route' a flight is servicing. A 'route' is a distinct set of Departure and Arrival airports, i.e LAX - JFK is a route.
* distance, the distance in KM between the Departure and Arrival airports
* days_to_nye, the number of days between New year's Eve and the flight
* days_to_ind,  the number of days between Independence Day and the flight

We used one-hot-encoding to represent the feature values as we observed that this method helped the prediction results.

# Feature importance

In the following graph feature importance for Random Forest Classifier is presented
![Feature importance RF](plots/fi_rf.png?raw=true "Feature importance RF")

In the following graph feature importance for XGB Classifier is presented
![Feature importance XGB](plots/fi_xgb.png?raw=true "Feature importance XGB")

# The model

In technical terms, we used scikitlearn framework to develop our model.
It is an ensemble model which contains the following classifiers:

1. A random forest classifier
2. A gradient bosting classifier
3. A bagging classifier made out of Decision Tree Classifiers
4. An XGB Classifier
5. An AdaBoost Classifier.

The results from the classifiers mentioned above are finally merged with a Voting Classifier that uses a soft voting policy.

Note: Due to the really small amount of data (just 8.900 training records) we decided **not to use neural networks** for this problem because they need more data to work correctly and provide reliable results.


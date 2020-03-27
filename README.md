# INF131: The case of flight passengers prediction
Build a classifier to predict number of passengers in flights

# Team members

Petros Demetrakopoulos 
Stavros Markopoulos
Nikos Kavalegas

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
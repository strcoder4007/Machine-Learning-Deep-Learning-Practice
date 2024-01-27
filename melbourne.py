import pandas as pd

melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

filtered_melbourne_data = melbourne_data.dropna(axis=0)
y = filtered_melbourne_data.Price

melbourne_features = ['Rooms', 'Bathroom', 'LandSize', 'YearBuilt']

X = filtered_melbourne_data[melbourne_features]

from sklearn.tree import DecisionTreeRegressor

melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(X, y)

from sklearn.metrics import mean_absolute_error


predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)

from sklearn.model_selection import train_test_split


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)


melbourne_model = DescisionTreeRegressor()
melbourne_model.fit(train_X, train_y)

predicted_home_prices = melbourne_model.predict(val_X)

print(mean_absolute_error(val_y, predicted_home_prices))

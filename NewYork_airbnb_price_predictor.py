import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

data = pd.read_csv("https://drive.google.com/uc?id=1Wa5ixqnsckyQQhb1Ap4wl37mSgnPI12F&export=download")

# to see columns and number of inputs per column
# data.info()

#drop the following columns: name, host_id, host_name, last_review, and reviews_per_month
data = data.drop(columns = ["name", "host_id", "host_name", "last_review", "reviews_per_month"], axis=1)
data.dropna(subset=['price'], inplace=True) # Essential for training

#display a summary of the statistics of the loaded data
#data.describe()

#plot boxplot for features: num_reviews, price, and availability_365
data.boxplot(column = ['number_of_reviews', 'price','availability_365'],figsize=(20,15))
plt.show()

#plot average price of a listing per neighbourhood_group
data.groupby("neighbourhood_group")['price'].mean().plot(kind='bar', figsize=(10, 6))
plt.title('Average Price by Neighbourhood Group')
plt.ylabel('Average Price ($)')
plt.xlabel('Neighbourhood Group')
plt.xticks(rotation=45)
plt.show()

#some assumptions that can be made at this point:
# location affects price
# entire homes have a higher price

#histogram of price by neighborhood to see price breakdown by range
for neighborhood in data["neighbourhood_group"].unique():
    plt.figure()
    data_tmp = data[(data["neighbourhood_group"] == neighborhood) & (data["price"] < 500)]
    data_tmp['price'].hist(bins=50)
    plt.title(f"Price Distribution in {neighborhood}")
    plt.xlabel("Price ($)")
    plt.ylabel("Number of Listings")
    plt.show()

#plot map of airbnbs throughout New York
# NOTE: This requires a 'newyork.png' file to be in your working directory.
try:
    newyork_img = mpimg.imread('newyork.png')
    ax = data.plot(kind="scatter", x="longitude", y="latitude",
                   s=data['price']/100, label="Price",
                   c="price", cmap=plt.get_cmap("jet"),
                   colorbar=False, alpha=0.4, figsize=(10,7))
    # overlay the New York map on the plotted scatter plot
    # note: plt.imshow still refers to the most recent figure
    # that hasn't been plotted yet.
    plt.imshow(newyork_img, extent=[-74.25, -73.7, 40.49, 40.92], alpha=0.5)
    plt.ylabel("Latitude", fontsize=14)
    plt.xlabel("Longitude", fontsize=14)

    # setting up heatmap colors based on median_house_value feature
    prices = data['price']
    tick_values = np.linspace(prices.min(), prices.max(), 11)
    cb = plt.colorbar()
    cb.ax.set_yticklabels([f"${int(v/1000)}k" for v in tick_values], fontsize=14)
    cb.set_label('Price', fontsize=16)

    plt.legend(fontsize=16)
    plt.savefig("newyork_housing_prices_plot.png")
    plt.show()
except FileNotFoundError:
    print("Skipping map plot because 'newyork.png' was not found.")


#Plot average price of room types who have availability greater than 180 days and neighbourhood_group is Manhattan
data_manhattan = data[data["neighbourhood_group"] == "Manhattan"]
data_manhattan_180 = data_manhattan[data_manhattan["availability_365"] > 180]
data_manhattan_180.groupby('room_type')['price'].mean().plot(kind='bar')
plt.title('Avg. Price in Manhattan for High-Availability Listings')
plt.ylabel('Average Price ($)')
plt.xlabel('Room Type')
plt.xticks(rotation=0)
plt.show()

#plotting correlation matrix
numeric_data = data.select_dtypes(include=[np.number])
corr_matrix_ny = numeric_data.corr()
print(corr_matrix_ny['price'].sort_values(ascending=False))

# The original code created and then immediately dropped these columns.
# This step is omitted in the corrected version for clarity, but your original comment is preserved here.
#I decided to drop this column (it was a column I augmented) because I saw that
# some of the data values in the column were NaN, inf, or 0. Also, inserting the
# median for empty data sets would not get rid of the inf or 0; the inf and 0
#may cause problems, which I don't want to risk.

#prepare the data using sk learn mixins
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Fill any remaining NaN values in numeric columns with the median
for col in data.select_dtypes(include=np.number).columns:
    median = data[col].median()
    data[col].fillna(median, inplace=True)

data_unlabeled = data.drop("price", axis=1) # drop labels for training set features
data_labels = data["price"].copy() # the input to the model should not contain the true label

# goal: housing_prepared of shape (N, D) and label_prepare of shape (N,)

# dealing with incomplete data
#incomplete data columns ("last review", "reviews_per_month") were dropped by
#instruction. There were no empty data sets when I checked.

# categorizal part
categorical_features = ["neighbourhood", "neighbourhood_group", "room_type"]
for c in categorical_features:
    le = preprocessing.LabelEncoder()
    data_unlabeled[c] = le.fit_transform(data_unlabeled[c])

# optionally: do some normalization here

# numerical preprocess
data_prepared = data_unlabeled.to_numpy()

# check the shape
print(f"Shape of prepared data: {data_prepared.shape}")
print(f"Shape of labels: {data_labels.shape}")

#Set aside 20% of the data as test test (80% train, 20% test).
train, test, target_train, target_test = train_test_split(data_prepared, data_labels, test_size=0.2, random_state=42)

#Predicting the price using MSE.
#Includes both test and train set MSE values.
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_reg = LinearRegression()
lin_reg.fit(train, target_train)

# Make predictions on the test set
test_pred = lin_reg.predict(test)

# let's try the full preprocessing pipeline on a few training instances
print("\nModel Evaluation:")
print("Predictions:", test_pred[:5])
print("Actual labels:", list(target_test)[:5])

mse = mean_squared_error(target_test, test_pred)
rmse = np.sqrt(mse)
print(f"\nRoot Mean Squared Error (RMSE) on Test Set: ${rmse:.2f}")
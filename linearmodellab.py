# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# %% [markdown]
## Question 1

# %%
# loading data

url = "https://raw.githubusercontent.com/DS3001/linearRegression/refs/heads/main/data/Q1_clean.csv"
df = pd.read_csv(url)
df.head()

# %%
# part 1 

df_averaged = df.groupby("Neighbourhood ")[["Price", "Review Scores Rating"]].mean()
# groups by the Neighborhood column first and then computes the average price and review scores rating for each neighborhood
# values in the price and review scores rating columns are the computed averages of each neighborhood
df_averaged = df_averaged.sort_values(by="Price", ascending=False)
# sorts the average prices of each of the neighborhoods by descending order
# the neighborhood on the top of the list has the highest average price 
df_averaged.head()

# Manhattan is the most expensive on average. 

# %%
# creating a kernel density plot of price grouping by neighborhood 

sns.kdeplot(df, x="Price", hue="Neighbourhood ")
plt.show()

# %%

df["log_price"] = np.log(df["Price"] + 1)
# calculates the log of price 
# add 1 to avoid log(0)

# creating a kernel density plot of log price grouping by neighborhood 
sns.kdeplot(df, x="log_price", hue="Neighbourhood ")
plt.show()

# %%
# part 2

df_without = pd.get_dummies(df, columns=["Neighbourhood "], prefix=["without"])
# one hot encoding the neighborhood column and adding the prefix "without" to the columns that are created 
df_without.head()

# %%

x_without = df_without.iloc[:, 5:10]
# choosing the neighborhood columns that have been one hot encoded 
y_without = df_without.iloc[:, 0]
# choosing the price column 

model_without = LinearRegression(fit_intercept=False).fit(x_without, y_without)
# fit_intercept argument specifies to create the linear regression object without an intercept 
# .fit applies the chosen columns into the linear regression model 

print(f"Without Intercept: Coefficient = {model_without.coef_}, R² = {model_without.score(x_without, y_without)}")

df_averaged.head()

# The coefficient values match the average prices of each neighborhood. 
# Each neighborhood gets its own coefficient to use for prediction. 
# The best coefficient value that will minimize error for prediction is the average price of that neighborhood 
# so the model will always predict that same value. 

# The coefficients in a regression of a continuous variable on one categorical variable 
# is the average of the categories for models without an intercept or the difference between 
# each average of the categories and the baseline category for models with intercepts.

# %%
# part 3 

df_with = pd.get_dummies(df, columns=['Neighbourhood '], drop_first=True, prefix=["with"])
df_with.head()

# drop_first=True drops one of the categories to remove redundancy. If you know that it's not any of the first 
# four neighbohoods, then you know that it must be the last neighborhood. Including all of the neighborhood categories 
# makes it redundant because the value for the last category can be inferred. This allows the model to estimate the 
# coefficients properly instead of incorporating redundant information.

# %%

x_with = df_with.iloc[:, 5:10]
y_with = df_with.iloc[:, 0]

model_with = LinearRegression(fit_intercept=True).fit(x_with, y_with)
# fit_intercept argument specifies to create the linear regression object with an intercept 

print(f"With Intercept: Coefficient = {model_with.coef_}, Intercept = {model_with.intercept_}, R² = {model_with.score(x_with, y_with)}")

df_averaged.head()

# The intercept is 75.27649769585331, which is the average price of the neighborhood that was dropped during pd.get_dummies. 
# The neighborhood that was removed becomes the baseline, making the intercept to be the average price of that neighborhood. 

# The coefficients are the difference between the average price of one neighborhood and the average price of the baseline category.

# The coefficients in part 2 can be derived by adding the new coefficients to the intercept. 

# %%
# part 4

features = ["Review Scores Rating", "with_Brooklyn", "with_Manhattan", "with_Queens", "with_Staten Island"]

x = df_with[features]
y = df_with["Price"]
# selecting the review scores rating and neighborhood columns to run a regression of price

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# test_size argument specifices the fraction of the test set 

model = LinearRegression().fit(x_train, y_train)

print(f"Coefficient: {model.coef_}")

y_pred = model.predict(x_test)
# model is the trained model with the coefficients
# .predict() applies the testing dataset to the model to get the predictions

mse = mean_squared_error(y_test, y_pred)
# calculates the mean squared error 
# measures average squared difference between predictions and true values
# unit is the same as the square of the target variable’s unit
rmse = np.sqrt(mse)
# calculates the root mean squared error 
# brings the error back to the same unit as the target variable, making it easier to interpret
r2 = r2_score(y_test, y_pred)
# calculates the R squared value 

print(f"Coefficient on Review Scores Rating: {model.coef_[0]}")
# model.coef_ is aligned with the order of columns in x,
# the review scores rating has an index of 0 because it's listed first in features 
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Manhattan is the most expensive neighborhood. 

# %%
# part 5

df_with_p = pd.get_dummies(df_with, columns=["Property Type"], drop_first=True, prefix=["with"])
df_with_p = df_with_p.drop(["Room Type", "log_price"], axis=1)

x_p = df_with_p.iloc[:, 1:]
# choosing the review scores rating, one hot encoded neighborhood and property type columns 

x_p_train, x_p_test, y_train, y_test = train_test_split(x_p, y, test_size=0.2, random_state=42)

model_p = LinearRegression().fit(x_p_train, y_train)

print(f"Coefficient: {model_p.coef_}")

y_p_pred = model_p.predict(x_p_test)

mse_p = mean_squared_error(y_test, y_p_pred)
rmse_p = np.sqrt(mse_p)
r2_p = r2_score(y_test, y_p_pred)

print(f"Coefficient on Review Scores Rating: {model_p.coef_[0]}")
print(f"RMSE: {rmse_p:.2f}")
print(f"R² Score: {r2_p:.4f}")

# The bungalow is the most expensive kind of property you can rent. 

# %%
# part 6 

# The coefficient on review scores rating changes from 1.2118517840632352 in part 4 
# to 1.2010106602298556 in part 5. 
# In multiple linear regression, each coefficient measures the effect of a variable while holding all other variables constant.
# By adding property type, some variation in price that was previously “attributed” to review scores is now explained by property type.
# As a result, the coefficient on review scores rating adjusts slightly downward to reflect its effect after controlling for neighborhood AND property type.

# %% [markdown]
## Question 2

# %%
# part 1

car = pd.read_csv("cars_hw.csv")
car.head()

# %%

car = car.drop("Unnamed: 0", axis=1)
# removes the "Unnamed: 0" column
car["log_price"] = np.log(car["Price"] + 1)
# creates a new column named "log_price" that holds the log values of the price column
car.head()

# %% 

car.boxplot(column="log_price")
# shows the outliers that needs to be addressed 

# %%

Q1 = car['log_price'].quantile(0.25)
Q3 = car['log_price'].quantile(0.75)
IQR = Q3 - Q1
car_clean = car[(car['log_price'] >= Q1 - 1.5*IQR) & (car['log_price'] <= Q3 + 1.5*IQR)]
# removing the outliers 

# %% 
# part 2

car_clean.groupby("Make")["Price"].describe()
# groups by car make to produce the descriptive statistics for price within each car make 

# %% 

sns.kdeplot(car_clean, x="Price", hue="Make")
plt.show()

# Jeep, Kia, and MG motors are the most expensive car brands. 
# These three brands are the only car companies that have a minmum car price in million dollars.  

# %%
# part 3 and 4 

# model of the numeric variables alone 
x_n = car_clean[["Make_Year", "Mileage_Run", "Seating_Capacity"]]
y = car_clean["log_price"]

x_n_train, x_n_test, y_train, y_test = train_test_split(x_n, y, test_size=0.2, random_state=42)

model_n = LinearRegression().fit(x_n_train, y_train)

y_pred_n = model_n.predict(x_n_test)

mse_n = mean_squared_error(y_test, y_pred_n)
rmse_n = np.sqrt(mse_n)
r2_n = r2_score(y_test, y_pred_n)

# %%

# model of the categorical variables alone 
car_clean_encoded = pd.get_dummies(car_clean, columns=["Make", "Color", "Body_Type", "No_of_Owners", "Fuel_Type", "Transmission", "Transmission_Type"], drop_first=True, prefix=["Make", "Color", "Type", "Owners", "Fuel", "Trans", "Trans_Type"])
car_clean_encoded = car_clean_encoded.drop(["Price", "log_price"], axis=1)

x_c = car_clean_encoded.iloc[:, 3:]

x_c_train, x_c_test, y_train, y_test = train_test_split(x_c, y, test_size=0.2, random_state=42)

model_c = LinearRegression().fit(x_c_train, y_train)

y_c_pred = model_c.predict(x_c_test)

mse_c = mean_squared_error(y_test, y_c_pred)
rmse_c = np.sqrt(mse_c)
r2_c = r2_score(y_test, y_c_pred)

# %% 

print(f"RMSE of Numerical: {rmse_n:.2f}")
print(f"R² Score of Numerical: {r2_n:.4f}")

print(f"RMSE of Categorical: {rmse_c:.2f}")
print(f"R² Score of Categorical: {r2_c:.4f}")

# The categorical model performed better on the test set
# because it has a lower RMSE score, but a higher R squared value. 

# %%

# model of the combined regressors 
x_a = pd.concat([car_clean[["Make_Year", "Mileage_Run", "Seating_Capacity"]], car_clean_encoded.iloc[:, 3:]], axis=1)
# pd.concat() combines multiple DataFrames (or Series) together by adding columns together

x_a_train, x_a_test, y_train, y_test = train_test_split(x_a, y, test_size=0.2, random_state=42)

model_a = LinearRegression().fit(x_a_train, y_train)

y_a_pred = model_a.predict(x_a_test)

mse_a = mean_squared_error(y_test, y_a_pred)
rmse_a = np.sqrt(mse_a)
r2_a = r2_score(y_test, y_a_pred)

print(f"RMSE of Combined: {rmse_a:.2f}")
print(f"R² Score of Combined: {r2_a:.4f}")

# The joint model performed better by about 40% than the numerical model and about 20% than the categorical model.

# %%
# part 5 

# model of raising numerical variables to powers
x_n_train, x_n_test, y_train, y_test = train_test_split(x_n, y, test_size=0.2, random_state=42)

results = {}
for degree in range(1, 11):
    # loops through degrees 1 through 10 
    pf  = PolynomialFeatures(degree=degree, include_bias=False)
    # creates the model object 
    # degree=degree argumuent controls the highest power the features can be raised to 
    # include_bias=False argument makes sure that a column of 1s are not added to represent the intercept
    # term because LinearRegression already accounts for it 
    Xtr = pf.fit_transform(x_n_train)
    # fits the training data and outputs a new feature matrix with polynomial terms
    # learns how to expand features and apply it
    Xte = pf.transform(x_n_test)
    # applies the same transformation to test data
    m = LinearRegression().fit(Xtr, y_train)
    # trains the linear regression model object using the feature matrix with polynomial terms 
    y_pred = m.predict(Xte)
    # uses the trained model to predict values for the test set
    r2  = m.score(Xte, y_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[f'degree_{degree}'] = {"R²": r2, "RMSE": rmse}
    print(f"Degree {degree}  |  R²: {r2:.4f}, RMSE: {rmse:.4f}")

# As degree increases, R squared increases then decreases
# RMSE initially decreases and then increases as degree increases. 
# At no point does the R squared value go negative on the test set. 
# Our best model with expanded features would be at degree 3 with R squared value of 0.4305 and RMSE of 0.3277. 
# This model does significantly worse than our joint model from part 4. 

# %%
# part 6 

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_a_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs. Predicted Values')
plt.show()

# The predicted values and true values roughly line up along the diagonal. 

# %% 

residual = y_test - y_a_pred
# residual value is the difference between the actual value and the predicted value

sns.kdeplot(residual)
plt.xlabel("Residual")
plt.show()

# The residual looks roughly bell shaped around zero. 

# Strengths:
# Our R-squared value is relatively high with roughly 77% of variability in the model explained.
# The kernel density plot of residuals also shows a roughly bell-shaped distribution around 0,
# which means that the model predictions are unbiased on average.
# Our RMSE is also on the lower end at 0.21.

# Weaknesses:
# We have several one-hot encoded features, which increased the number of predictors significantly. 
# This means that the model might not generalize as well, and that our model could be overfitted.
# While the residuals are roughly bell-shaped, there may still be outliers or extreme residuals, 
# which is indicated in the tail in the model. This means that the model may struggle with 
# predicting very expensive or very cheap cars.

# %% [markdown]
## Question 3

# %%
# part 1

# dataset from https://www.geeksforgeeks.org/machine-learning/dataset-for-linear-regression/
# "This dataset includes information about medical charges billed by health insurance companies 
# with features like age, sex, BMI, children, smoker status, region and the charges billed."
insurance = pd.read_csv("insurance.csv")
insurance.head()

# %%
# part 2

insurance["region"] = insurance["region"].astype("category")
# converting the region column into a categorical data type 

# %%

sns.kdeplot(insurance, x="charges", hue="region")
plt.show()

# %% 

insurance["log_charges"] = np.log(insurance["charges"] + 1)
# taking the log of charges because it's skewed to the right 

sns.kdeplot(insurance, x="log_charges", hue="region")
plt.show()

# %% 

# one hot encoding the regions column 
insurance_region = pd.get_dummies(insurance, columns=["region"], drop_first=True, prefix=["region"])
insurance_region.head()

# %%

insurance.isna().sum()
# checking for null values 

# %% 

insurance.boxplot(column="log_charges")
# checking to see if there are any outliers 

# %% 

insurance.groupby("region")["charges"].describe()
# grouping by region and producing the descriptive statistics of charges for each region

# %%

# features/predictors : age, bmi, region
# target/outcome variable : medical charges billed by health insurance companies (charges)

# %%
# part 3 and 4 

# model of the numeric variables alone 
x_n_i = insurance_region[["age", "bmi"]]
y = insurance_region["log_charges"]

x_i_train, x_i_test, y_train, y_test = train_test_split(x_n_i, y, test_size=0.2, random_state=42)

model_n_insurance = LinearRegression().fit(x_i_train, y_train)

y_pred_n_insurance = model_n_insurance.predict(x_i_test)

mse_n_insurance = mean_squared_error(y_test, y_pred_n_insurance)
rmse_n_insurance = np.sqrt(mse_n_insurance)
r2_n_insurance = r2_score(y_test, y_pred_n_insurance)

# %%

# model of the categorical variables alone 
x_c_i = insurance_region[["region_northwest", "region_southeast", "region_southwest"]]

x_c_i_train, x_c_i_test, y_train, y_test = train_test_split(x_c_i, y, test_size=0.2, random_state=42)

model_c_insurance = LinearRegression().fit(x_c_i_train, y_train)

y_c_pred_insurance = model_c_insurance.predict(x_c_i_test)

mse_c_insurance = mean_squared_error(y_test, y_c_pred_insurance)
rmse_c_insurance = np.sqrt(mse_c_insurance)
r2_c_insurance = r2_score(y_test, y_c_pred_insurance)

# %% 

print(f"RMSE of Numerical: {rmse_n_insurance:.2f}")
print(f"R² Score of Numerical: {r2_n_insurance:.4f}")

print(f"RMSE of Categorical: {rmse_c_insurance:.2f}")
print(f"R² Score of Categorical: {r2_c_insurance:.4f}")

# %%

# model of the combined regressors 
x_a_i = insurance_region[["age", "bmi", "region_northwest", "region_southeast", "region_southwest"]]

x_a_i_train, x_a_i_test, y_train, y_test = train_test_split(x_a_i, y, test_size=0.2, random_state=42)

model_a_insurance = LinearRegression().fit(x_a_i_train, y_train)

y_a_pred_insurance = model_a_insurance.predict(x_a_i_test)

mse_a_insurance = mean_squared_error(y_test, y_a_pred_insurance)
rmse_a_insurance = np.sqrt(mse_a_insurance)
r2_a_insurance = r2_score(y_test, y_a_pred_insurance)

print(f"RMSE of Combined: {rmse_a_insurance:.2f}")
print(f"R² Score of Combined: {r2_a_insurance:.4f}")

# %%
# part 5 

# The numerical model performed the best. 
# The numiercal and joint model produced the same RMSE values of 0.77 and similar R squared 
# values of 0.3431 and 0.3458 respectively. This shows that adding categorical regressors 
# have very little effect and that most of the predictive power comes from numerical features. 
# The categorical model performed poorly as it has an R squared value near 0 and RMSE score of 0.95,
# suggesting that categorical features alone do not effectively explain variation in the target variable.

# %% 
# part 6 

# I learned that not all variables are strong predictors of the target variable. 
# The categorical features provided almost no predictive power, 
# while the numerical features explained a moderate amount of variation. 
# Additionally, combining features only slightly improved performance, 
# indicating that adding more variables does not necessarily lead to better models 
# unless they contain meaningful information. Overall, this highlights that 
# identifying useful relationships in real-world datasets can be challenging.

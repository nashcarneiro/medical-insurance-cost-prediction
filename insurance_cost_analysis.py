# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# %%
df = pd.read_csv("medical_insurance_dataset.csv", header=None)
print(df.head(5))

# %%
# CSV is without header names so lets add headers into out dataframe
df.columns=["age", "gender", "bmi", "no_of_children", "smoker", "region", "charges"]
print(df.head(5))

# %%
# Checking for missing values
df.replace('?', np.nan, inplace=True)
print(df.info())

# %%
mean_age = df['age'].astype('float').mean()
df['age'] = df['age'].replace(np.nan, mean_age)
smoker_mode = df['smoker'].value_counts().idxmax()
df['smoker'] = df['smoker'].replace(np.nan, smoker_mode)
print(df.info())

# %%
# Round up charges
df['charges'] = np.round(df['charges'], 2)
print(df.head())

# %%
# Exploratory Data Analysis
sns.regplot(x='bmi', y='charges', data=df)
plt.ylim(0,)
plt.show()
sns.boxplot(x='smoker', y='charges', data=df)
plt.show()
charges_corr = (
    df[['age', 'bmi', 'no_of_children', 'smoker', 'region']]
    .corrwith(df['charges'])
    .sort_values()
)
print("Correlation of Insurance Features with Charges:\n",charges_corr)

plt.figure(figsize=(10,6))
sns.barplot(
    x=charges_corr.values,
    y=charges_corr.index
)
plt.title("Correlation of Insurance Features with Charges")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Insurance Features")
plt.tight_layout()
plt.show()


# %%
# Model Development
# Smoker shows highest correlation to charges
x_data = df[['smoker']]
y_data = df['charges']
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1)
lm = LinearRegression()
lm.fit(x_train, y_train)
linear_regression_score = lm.score(x_test, y_test)
print("Score:", linear_regression_score)

# %%
# Lets also use other attributes along with smoke to enhance the performance of the model
z_data = df[['age', 'bmi', 'no_of_children', 'smoker', 'region']]
x_train, x_test, y_train, y_test = train_test_split(z_data, y_data, test_size=0.2, random_state=1)
lm.fit(x_train, y_train)
multiple_linear_regression_score = lm.score(x_test,y_test)
print("Score:", multiple_linear_regression_score)

# %%
# Lets create a pipeline that uses standard scaler, polynomial features and linear regression to create a model that can predict charges using the attributes in the dataset.

input = [('scale',StandardScaler()),('polynomial',PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe = Pipeline(input)
pipe.fit(x_train, y_train)
ypipe = pipe.predict(x_test)
polynomial_pipeline_score = pipe.score(x_test, y_test)
print("Score:", polynomial_pipeline_score)

# %%
# Model refinement
rm = Ridge(alpha=0.1)
rm.fit(x_train, y_train)
yhat = rm.predict(x_test)
ridge_score = rm.score(x_test, y_test)
print("Score:", ridge_score)

# %%
pf = PolynomialFeatures(degree=2)
x_train_pf = pf.fit_transform(x_train)
x_test_pf = pf.transform(x_test)
rm.fit(x_train_pf, y_train)
yhat = rm.predict(x_test_pf)
polynomial_ridge_score = rm.score(x_test_pf, y_test)
print("Score:", polynomial_ridge_score)


#Lets do a model comparison now
models_score = {
    "Linear Regression":linear_regression_score,
    "Multiple Linear Regression":multiple_linear_regression_score,
    "Polynomial Regression Pipeline":polynomial_pipeline_score,
    "Ridge Regression":ridge_score,
    "Polynomial Ridge Regression":polynomial_ridge_score
}

score_df = pd.DataFrame(
    list(models_score.items()),
    columns=['Models', 'R2 Score']
)

sns.barplot(
    x='R2 Score',
    y='Models',
    data=score_df
)

plt.xlabel("R2 Score")
plt.ylabel("Models")
plt.tight_layout()
plt.show()

# Looking at this barplot, we see that both the Polynomial Rergressions with pipeline and ridge regression have the highest scores, so well visualize the actual vs predicted values for it.

plt.figure(figsize=(10,6))
sns.scatterplot(
    x=y_test,
    y=ypipe,
     color='steelblue',
)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle='--',
    color="red",
    label="Perfect Prediction"
)

plt.legend()
plt.title("Actual vs Predicted Medical Insurance Charges — Polynomial Regression Pipelines")
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.tight_layout()
plt.show()
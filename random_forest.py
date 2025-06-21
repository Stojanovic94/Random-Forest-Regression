# 1. Import libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load Kaggle dataset
import kagglehub
import pandas as pd
import os

# Download latest version
path = kagglehub.dataset_download("altavish/boston-housing-dataset")
housing_csv = os.path.join(path, "HousingData.csv")
print("Path to dataset files:", path)
df_housing = pd.read_csv(housing_csv)
print(df_housing)

# 3. Prepare data
data = pd.read_csv(housing_csv)

# Convert all column names to lowercase
data.columns = data.columns.str.lower()

# Define features and target variable
X = data.drop("medv", axis=1)
y = data["medv"]

# 4. Categorical and numerical columns
categorical_features = ['chas']
numerical_features = [col for col in X.columns if col not in categorical_features]

# Preprocessor (no scaling needed for Random Forest)
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(), categorical_features)
], remainder='passthrough')

# 5. Create pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# 6. Train the model
model.fit(X_train, y_train)

# 7. Prediction and evaluation
predictions = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, predictions))

# 8. Feature importance analysis
feature_names = list(model.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out()) + numerical_features
importances = model.named_steps['regressor'].feature_importances_
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# 9. Visualization
sns.barplot(x="Importance", y="Feature", data=importance_df)
plt.title("Feature Importance in Random Forest Model")
plt.tight_layout()
plt.show()
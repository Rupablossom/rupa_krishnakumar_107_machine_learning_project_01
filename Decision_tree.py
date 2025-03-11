from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv("rupa_107_machine_learning_project.csv")

# Drop missing values
df = df.dropna()

# Convert categorical columns to numerical values
label_encoder = LabelEncoder()
df['Specialization'] = label_encoder.fit_transform(df['Specialization'])
df['Items'] = label_encoder.fit_transform(df['Items'])
df['ProcedureName'] = label_encoder.fit_transform(df['ProcedureName'])

# Define Features (X) and Target Variable (y)
X = df[['Specialization', 'Items', 'ProcedureName']]  # Input Features
y = df['Amount']  # Target Variable (Medical Cost)

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Ensure cv is at least 2
cv_value = max(2, min(3, df['Specialization'].value_counts().min()))

# Set up the parameter grid for hyperparameter tuning
param_grid = {
    'criterion': ['squared_error', 'friedman_mse'],
    'max_depth': [2, 3, 4, 5, None],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize a Decision Tree model (for Regression)
tree_reg = DecisionTreeRegressor(random_state=42)

# Initialize GridSearchCV with adjusted cv
grid_search = GridSearchCV(
    estimator=tree_reg,
    param_grid=param_grid,
    cv=cv_value,  # Ensured cv ≥ 2
    scoring='r2'
)

# Perform Grid Search on the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Hyperparameters:", best_params)
print("Best Model:", best_model)

# Visualize the Decision Tree
plt.figure(figsize=(15, 8))
tree.plot_tree(best_model, feature_names=['Specialization', 'Items', 'ProcedureName'], filled=True)
plt.show()

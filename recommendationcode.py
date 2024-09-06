# Step 1 importing the required libraries I will be using
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

file_path = r"C:\Users\8noor\Desktop\Seerat\Recommendation\ratings.csv"
ratings = pd.read_csv(file_path)

# Checking the first few rows of the dataset to confirm it is loaded correctly
print(ratings.head())

reader = Reader(rating_scale=(0.5, 5.0))  

data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Split the data into training and testing sets (75% training, 25% testing)
trainset, testset = train_test_split(data, test_size=0.25)
model = SVD()

# Train the model on the training set
model.fit(trainset)

# Make predictions on the testset
predictions = model.test(testset)

# Evaluate the model using RMSE (Root Mean Squared Error)
rmse = accuracy.rmse(predictions)

# Output the RMSE value
print(f"RMSE: {rmse}")

# the RMSE is 0.8750 now to improve the RMSE score, tune the hyperparameters
from surprise.model_selection import GridSearchCV

# Defining the parameter grid to search
param_grid = {
    'n_factors': [50, 100, 150],  # Number of latent factors
    'n_epochs': [20, 30, 50],     # Number of training epochs
    'lr_all': [0.002, 0.005, 0.01],  # Learning rate
    'reg_all': [0.02, 0.1, 0.4]   # Regularization term
}

# Perform grid search with cross-validation
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
gs.fit(data)

# Output the best score and parameters
print(f"Best RMSE score: {gs.best_score['rmse']}")
print(f"Best hyperparameters: {gs.best_params['rmse']}")


# the RMSE is 0.8764 now to improve the RMSE score, tune the hyperparameters

from surprise.model_selection import GridSearchCV

# Define the parameter grid to search over
param_grid = {
    'n_factors': [50, 100, 150],  # Latent factors
    'n_epochs': [20, 30, 40],     # Epochs for training
    'lr_all': [0.002, 0.005, 0.01],  # Learning rate
    'reg_all': [0.02, 0.1, 0.4]   # Regularization
}

# Use GridSearchCV to search for the best parameters
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
gs.fit(data)

# Output the best score and parameters
print(f"Best RMSE score: {gs.best_score['rmse']}")
print(f"Best hyperparameters: {gs.best_params['rmse']}")
#RMSE now is 0.8816
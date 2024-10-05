import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error)

from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

def load_data(train_path:str, test_path:str) -> (pd.DataFrame, pd.DataFrame):
    """
    Loads the train and test data into pandas DataFrames
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def preprocess_data(train, test):
    train_cols = [
        col for col in train.columns if col not in ['id', 'price']
    ]

    categorical_cols = ["type", "sector"]
    target = "price"

    categorical_transformer = TargetEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical',
             categorical_transformer,
             categorical_cols)
        ])

    return train, test, train_cols, target, preprocessor

def build_and_train_model(train, train_cols, target, preprocessor):
    steps = [
        ('preprocessor', preprocessor),
        ('model', GradientBoostingRegressor(**{
            "learning_rate": 0.01,
            "n_estimators": 300,
            "max_depth": 5,
            "loss": "absolute_error"
        }))
    ]

    pipeline = Pipeline(steps)

    pipeline.fit(train[train_cols], train[target])

    return pipeline


def evaluate_model(pipeline, test, train_cols, target):
    '''
    Evaluates the model on the test set and prints metrics.
    '''
    test_predictions = pipeline.predict(test[train_cols])
    test_target = test[target].values

    print("RMSE: ", np.sqrt(mean_squared_error(test_predictions, test_target)))
    print("MAPE: ", mean_absolute_percentage_error(test_predictions, test_target))
    print("MAE : ", mean_absolute_error(test_predictions, test_target))

# Save the trained model to a pickle file
def save_model():
    model = build_and_train_model(train, train_cols, target, preprocessor)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    # Paths can be parameterized
    # Load data
    train, test = load_data('train.csv', 'test.csv')

    # Preprocess data
    train, test, train_cols, target, preprocessor = preprocess_data(train, test)

    # Train model
    pipeline = build_and_train_model(train, train_cols, target, preprocessor)

    # Evaluate model
    evaluate_model(pipeline, test, train_cols, target)
    
    #Save model
    save_model()
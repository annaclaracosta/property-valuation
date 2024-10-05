
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from sklearn.ensemble import GradientBoostingRegressor

# Function to load data
def load_data(train_path: str):
    return pd.read_csv(train_path)

# Train your model (adjust accordingly to your data)
def train_model():
    # Load data (assuming train.csv is available in the same directory)
    train = load_data('train.csv')
    
    # Feature engineering
    train_cols = [col for col in train.columns if col not in ['id', 'price']]
    categorical_cols = ["type", "sector"]
    target = "price"

    categorical_transformer = TargetEncoder()
    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', categorical_transformer, categorical_cols)
        ])

    steps = [
        ('preprocessor', preprocessor),
        ('model', GradientBoostingRegressor(learning_rate=0.01, n_estimators=300, max_depth=5, loss="absolute_error"))
    ]

    pipeline = Pipeline(steps)
    pipeline.fit(train[train_cols], train[target])
    
    return pipeline

# Save the trained model to a pickle file
def save_model():
    model = train_model()
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    save_model()

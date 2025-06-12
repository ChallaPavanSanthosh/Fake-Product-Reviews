import os
import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            print("Loading model and preprocessor...")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("Successfully loaded model and preprocessor")

            print("Transforming input features...")
            
            data_scaled = preprocessor.transform(features)

            print("Predicting...")
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, 
                 review_text: str,
                 product_category: str,
                 user_rating: float,
                 verified_purchase: str,
                 review_length: int):
        self.review_text = review_text
        self.product_category = product_category
        self.user_rating = user_rating
        self.verified_purchase = verified_purchase
        self.review_length = review_length

    def get_data_as_data_frame(self):
        try:
            input_dict = {
                "review_text": [self.review_text],
                "product_category": [self.product_category],
                "user_rating": [self.user_rating],
                "verified_purchase": [self.verified_purchase],
                "review_length": [self.review_length]
            }

            return pd.DataFrame(input_dict)

        except Exception as e:
            raise CustomException(e, sys)

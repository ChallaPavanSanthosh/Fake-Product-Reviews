import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def add_manual_features(self, df):  
        df["review_length"] = df["text_"].apply(len)
        df["caps_count"] = df["text_"].apply(lambda x: sum(1 for c in x if c.isupper()))
        df["exclaim_count"] = df["text_"].apply(lambda x: x.count("!"))
        return df
    #
    def get_data_transformer_object(self):
        '''
        This function is responisble for data transformation
        '''
        try:
            numerical_columns = ["rating", "review_length", "caps_count", "exclaim_count"] #
            categorical_columns = ["category"]
            text_column = "text_"

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))
            ])

            text_pipeline = Pipeline(steps=[
                ("selector", FunctionTransformer(lambda x: x["text_"], validate=False)),
                # ("tfidf", TfidfVectorizer(max_features=1000, stop_words="english"))
                ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english"))

            ])

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"text columns:{text_column}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_columns),
                    ("cat", cat_pipeline, categorical_columns),
                    ("txt", text_pipeline, ["text_"])
                ],
                remainder='drop'
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Add manual features
            train_df = self.add_manual_features(train_df) #
            test_df = self.add_manual_features(test_df) #

            logging.info("Manual features added") #

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "label"

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            label_encoder = LabelEncoder()
            target_feature_train_df = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_df = label_encoder.transform(target_feature_test_df)

            for index, class_label in enumerate(label_encoder.classes_):
                logging.info(f"Label encoding: {class_label} --> {index}")


            logging.info("Applying preprocessing object on training and testing data")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Convert sparse matrix to dense if needed
            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = input_feature_test_arr.toarray()


            print("input_feature_train_arr shape:", input_feature_train_arr.shape)
            print("target_feature_train_df shape:", np.array(target_feature_train_df).shape)

            # target_feature_train_df = target_feature_train_df.values.reshape(-1, 1)
            # target_feature_test_df = target_feature_test_df.values.reshape(-1, 1)

            target_feature_train_df = np.array(target_feature_train_df).reshape(-1, 1)
            target_feature_test_df = np.array(target_feature_test_df).reshape(-1, 1)


            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

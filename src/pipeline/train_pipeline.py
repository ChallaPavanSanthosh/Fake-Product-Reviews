import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion
from src.exception import CustomException
from src.logger import logging

def main():
    try:
        logging.info("Starting training pipeline...")

        # 1. Data Ingestion
        ingestion = DataIngestion()
        train_data_path, test_data_path = ingestion.initiate_data_ingestion()

        # 2. Data Transformation
        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(train_data_path, test_data_path)

        # 3. Model Training
        trainer = ModelTrainer()
        final_accuracy = trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info(f"Training pipeline completed. Final accuracy: {final_accuracy:.4f}")
        print(f"\n🎯 Final model test accuracy: {final_accuracy:.4f}")

    except Exception as e:
        logging.error("Training pipeline failed.")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
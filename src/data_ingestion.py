import os
import pandas as pd 
from google.cloud import storage
from sklearn.model_selection import train_test_split

from src.custom_exception import CustomException
from src.logger import get_loggers

from config.path_config import *

from utils.common_functions import read_yaml


logger = get_loggers(__name__)

class DataIngestion:
    def __init__(self,config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.bucket_file_name = self.config["bucket_file_name"]
        self.train_test_ratio = self.config["train_ratio"]
        os.makedirs(RAW_DIR,exist_ok=True)
        logger.info(f"Data Ingestion Started bucket name : {self.bucket_name} file {self.bucket_file_name}")

    def download_csv_from_gcp(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.bucket_file_name)

            blob.download_to_filename(RAW_FILE_PATH)
            logger.info(f"CSV File is Successfully downloaded to {RAW_FILE_PATH}")
        except Exception as e:
            logger.error("Error while downloading csv file")
            raise CustomException("Failed to download the csv ",e)

    def split_data(self):
        try:
            logger.info("starting split process")
            data = pd.read_csv(RAW_FILE_PATH)
            train,test = train_test_split(data,test_size=1-self.train_test_ratio,random_state=42)
            train.to_csv(TRAIN_FILE_PATH)
            test.to_csv(TEST_FILE_PATH)
            logger.info("Data saved to train & test directory")
        except Exception as e:
            logger.error("Error while spliting csv file")
            raise CustomException("Failed to split the csv ",e)

    def run(self):
        try:
            logger.info("Started Run Method")

            self.download_csv_from_gcp()
            self.split_data()
            
            logger.info("Data ingestion is completd")
        except CustomException as e:
            logger.error(f"Error in run method {str(e)}")

        finally:
            logger.info("data ingestion completed")


if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()
import os
import pandas as pd 

from src.custom_exception import CustomException
from src.logger import get_loggers

import yaml

logger = get_loggers(__name__)


def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError("File is not there in the given path")
        with open(file_path,"r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info(f"config file : {config}")
            logger.info("Succefully read the YAML File")
            return config
    except Exception as e:
        logger.error("Error while reading YAML file")
        raise CustomException("Failed to read the YAML File", e)


def load_data(path):
    try:
        logger.info("Loading Data to DataFrame")
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error loading the data {e}")
        raise CustomException("Failed to load data", e)

    
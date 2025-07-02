import os
import pandas as pd 
import numpy as np 
from src.logger import get_loggers
from src.custom_exception import CustomException
from config.path_config import *
from utils.common_functions import read_yaml,load_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


logger = get_loggers(__name__)

class DataProcessor:
    def __init__(self,train_path,test_path,processd_dir,config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processd_dir = processd_dir
        self.config = read_yaml(config_path)
        if not os.path.exists(self.processd_dir):
            os.makedirs(self.processd_dir)
        

    def preprocess_data(self,df):
        try:
            logger.info("Starting data processing step")
            logger.info("Droping Columns")
            df.drop(columns=['Unnamed: 0','Booking_ID'],inplace=True)
            logger.info("Droping Duplicates")
            df.drop_duplicates(inplace=True)
            cat_cols = self.config['data_processing']['categorical_columns']
            num_cols = self.config['data_processing']['numerical_columns']
            logger.info("Label Encoding Started")
            label_encoder = LabelEncoder()
            mappings = {}
            for column in cat_cols:
                df[column] = label_encoder.fit_transform(df[column])
                mappings[column] = {label:code for label,code in zip(label_encoder.classes_,label_encoder.transform(label_encoder.classes_))}
            logger.info(f"label mappings : {mappings}")
            logger.info("Removing Skewed data")
            skewness = df.skew()
            for column in df.columns:
                if skewness[column] > self.config['data_processing']['skewness_threshold']:
                    df[column] = np.log1p(df[column])
            return df
        except Exception as e:
            logger.error(f"Error during preprocessing stage {e}")
            raise CustomException("Error while preprocess data ",e)
        
    def balance_data(self,df):
        try:
            logger.info("Fixing imbalanced dataset")
            X = df.drop(columns=['booking_status'])
            y = df['booking_status']
            smote = SMOTE(random_state=42)
            X_res,y_res = smote.fit_resample(X,y)
            balanced_df = pd.DataFrame(X_res,columns=X.columns)
            balanced_df["booking_status"] = y_res
            logger.info("Fixed Imbalanced dataset")
            return balanced_df
        except Exception as e:
            logger.error(f"Error during balancing dataset {e}")
            raise CustomException("Error while balancing dataset ",e)

    def feature_selection(self,df):
        try:
            logger.info("Starting Feature Engineering")
            X = df.drop(columns=['booking_status'])
            y = df['booking_status']
            model = RandomForestClassifier(random_state=42)
            model.fit(X,y)
            feature_importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                                                    'feature':X.columns,
                                                    'importance':feature_importance
                                                })
            top_features_importance_df = feature_importance_df.sort_values(by="importance",ascending=False)
            n_features = self.config['data_processing']['top_n_feature']
            top_n_features = top_features_importance_df["feature"].head(n_features).values
            top_n_df = df[top_n_features.tolist() + ['booking_status']]
            logger.info("Feature Selection Completed")
            logger.info(f"Top {n_features}s : {top_n_df}")
            return top_n_df
        except Exception as e:
            logger.error(f"Error during feature selection {e}")
            raise CustomException("Error while feature selection stage",e)

    def save_data(self,df,filepath):
        try:
            logger.info("Saving data to Dir")
            df.to_csv(filepath,index=False)
            logger.info("Saved data")
        except Exception as e:
            logger.error(f"Error during saving data {e}")
            raise CustomException("Error while saving data",e)
    
    def process(self):
        try:
            logger.info("Loading Data from RAW DIR")
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)

            train_df = self.feature_selection(train_df)
            test_df = test_df[train_df.columns]

            self.save_data(train_df,PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df,PROCESSED_TEST_DATA_PATH)

            logger.info("Data processing is completed")

        except Exception as e:
            logger.error(f"Error during preprocessing data {e}")
            raise CustomException("Error while preprocessing data",e)


if __name__ == '__main__':
    processor = DataProcessor(TRAIN_FILE_PATH,TEST_FILE_PATH,PROCESSED_DIR,CONFIG_PATH)
    processor.process()

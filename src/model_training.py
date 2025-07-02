import os
import pandas as pd 
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from src.logger import get_loggers
from src.custom_exception import CustomException
from config.path_config import *
from config.model_params import *
from utils.common_functions import read_yaml,load_data
from scipy.stats import randint

import mlflow
import mlflow.sklearn

logger = get_loggers(__name__)


class ModelTraining:
    def __init__(self,train_path,test_path,model_output_path):
        self.train_path = train_path,
        self.test_path = test_path,
        self.model_output_path = model_output_path
        self.params_dist = LIGHTLGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS


    def load_and_split(self):
        try:
            logger.info(f"loading data from {self.train_path}")
            train_df = load_data(self.train_path[0])
            logger.info(f"loading data from {self.test_path}")
            test_df = load_data(self.test_path[0])
            X_train = train_df.drop(columns=['booking_status'])
            y_train = train_df['booking_status']
            X_test = test_df.drop(columns=['booking_status'])
            y_test = test_df['booking_status']
            logger.info('Spliting Completed')
            return X_train,X_test,y_train,y_test
        except Exception as e:
            logger.error(f"Error in load and split method {e}")
            raise CustomException("Error in load and split method",e)

    def train_lgbm(self,X_train,y_train):
        try:
            logger.info('Initializing our model')
            lgbm_model = lgb.LGBMClassifier(random_state=42)
            logger.info("Starting Hyperparameter Tuning ")
            rand_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params["n_iter"],
                cv=self.random_search_params["cv"],
                n_jobs=self.random_search_params["n_jobs"],
                random_state=self.random_search_params["random_state"],
                scoring=self.random_search_params["scoring"]
            )
            logger.info("Staring hyper parameter tuning")
            rand_search.fit(X_train,y_train)
            logger.info("Tuning Completed")
            best_params = rand_search.best_params_
            best_lgbm = rand_search.best_estimator_
            logger.info(f'Best Model Params {best_params}')
            return best_lgbm;
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning {e}")
            raise CustomException("Error in hyperparameter tuning method",e)

    def evaluate(self,model,X_test,y_test):
        try:
            logger.info('Evaluation Started')
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test,y_pred)
            precsion = precision_score(y_test,y_pred)
            recall = recall_score(y_test,y_pred)
            f1 = f1_score(y_test,y_pred)
            logger.info(f"Accuracy Score : {accuracy}")
            logger.info(f"Precision Score : {precsion}")
            logger.info(f"Recall Score : {recall}")
            logger.info(f"F1 Score : {f1}")
            return {
                'accuracy score':accuracy,
                'precision score':precsion,
                'recall score':recall,
                'f1 score':f1
            }
        except Exception as e:
            logger.error(f"Error in Evaluation {e}")
            raise CustomException("Error in Evaluation method",e)

    
    def save_model(self,model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path),exist_ok=True)
            logger.info("Saving Model")
            joblib.dump(model,self.model_output_path)
            logger.info(f'Model Saved to {self.model_output_path}')
        except Exception as e:
            logger.error(f"Error in Save Model method {e}")
            raise CustomException("Error in Save Model method",e)
        
    def run(self):
        try:
            with mlflow.start_run():
                logger.info('Starting model Training pipeline')
                logger.info('Starting MLFLOW Expiramentaion')
                logger.info('Logging the training and testing datatset to mlflow')
                mlflow.log_artifact(self.train_path[0],artifact_path='datasets')
                mlflow.log_artifact(self.test_path[0],artifact_path='datasets')
                X_train,X_test,y_train,y_test = self.load_and_split()
                best_lgbm_model = self.train_lgbm(X_train,y_train)
                metrics = self.evaluate(best_lgbm_model,X_test,y_test)
                self.save_model(best_lgbm_model)
                logger.info('logging model to mlflow')
                mlflow.log_artifact(self.model_output_path)
                logger.info('logging model params to mlflow')
                mlflow.log_params(best_lgbm_model.get_params())
                logger.info('logging model metrics to mlflow')
                mlflow.log_metrics(metrics)
                logger.info('Model Training Completed')
        except Exception as e:
            logger.error(f"Error in Model Train Pipeline  {e}")
            raise CustomException("Error in Model Train Pipeline",e) 


if __name__ == '__main__':
    trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH,PROCESSED_TEST_DATA_PATH,MODEL_OUTPUT_PATH)
    trainer.run()
    
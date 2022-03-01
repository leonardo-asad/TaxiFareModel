# imports
from TaxiFareModel.data import get_data
from TaxiFareModel.data import clean_data

from TaxiFareModel.encoders import DistanceTransformer
from TaxiFareModel.encoders import TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property

import joblib


MLFLOW_URI = "https://mlflow.lewagon.co/"

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = "[AR] [BUENOS AIRES] [LeoA] TaxiFareModel v1"


    def set_pipeline(self, estimator):
        """defines the pipeline as a class attribute"""
        # Preprocesing Pipeline
        # Distance preprocessing
        dist_pipe = make_pipeline(DistanceTransformer(), StandardScaler())

        # Time Features Pipeline
        time_pipe = make_pipeline(TimeFeaturesEncoder('pickup_datetime'),
                                  OneHotEncoder(handle_unknown='ignore'))

        # Preprocessing pipeline
        preproc_pipe = make_column_transformer(
            (dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            (time_pipe, ['pickup_datetime'])
        )

        # Model Pipeline
        pipe = make_pipeline(preproc_pipe, estimator)

        # Define the pipeline object as a class atribute
        self.pipeline = pipe


    def run(self, estimator=LinearRegression()):
        """set and train the pipeline"""
        self.set_pipeline(estimator)
        self.pipeline.fit(self.X, self.y)
        return self.pipeline



    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse

    def cross_validate(self, estimator=LinearRegression(), scoring='neg_root_mean_squared_error'):
        self.set_pipeline(estimator)
        res = cross_val_score(estimator=self.pipeline,
                              X=self.X,
                              y=self.y,
                              scoring=scoring,
                              cv=5,
                              n_jobs=-1)
        return res.mean()


    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, 'model.joblib')


if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    y = df.pop('fare_amount')
    X = df
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # train
    # Instanciate the class
    trainer = Trainer(X_train, y_train)
    #  Train the pipeline
    trainer.run()
    # evaluate
    rmse = trainer.evaluate(X_test, y_test)

    # Log estimator and rmse
    trainer.mlflow_log_metric('rmse', rmse)
    trainer.mlflow_log_param('estimator', 'Linear Regression')

    # Retrive the id to find the experiment
    experiment_id = trainer.mlflow_experiment_id

    # Save model
    trainer.save_model()

    print(f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}")

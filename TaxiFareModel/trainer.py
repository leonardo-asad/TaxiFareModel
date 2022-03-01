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



class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
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
        pipe = make_pipeline(preproc_pipe, LinearRegression())

        # Define the pipeline object as a class atribute
        self.pipeline = pipe


    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse

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
    # Call the set pipeline method
    #trainer.set_pipeline()
    #  Train the pipeline
    trainer.run()
    # evaluate
    rmse = trainer.evaluate(X_test, y_test)
    print(rmse)

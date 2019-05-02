#!/usr/bin/env python
# -*- coding: utf-8 -*-

# We need sys to get the command line arguments
import sys

from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.recommendation import ALS


def main(spark, data_file, model_file):
    '''
    Parameters
    ----------
    spark : SparkSession object
    data_file : string, path to the parquet file to load
    model_file : string, path to store the serialized model file
    '''

    # Load the parquet file
    train = spark.read.parquet(data_file)
    
    indexer_user = StringIndexer(inputCol="user_id", outputCol="user", handleInvalid="skip")
#     indexer_item = StringIndexer(inputCol="track_id", outputCol="item", handleInvalid="skip")
    
    als = ALS(userCol="user", itemCol="item", ratingCol="count")
    pipeline = Pipeline(stages = [indexer_user, indexer_item, als])
    model = pipeline.fit(train)
    model.save(model_file)
    
    

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('recommender_train').getOrCreate()

    # Get the filename from the command line
    data_file = sys.argv[1]

    # And the location to store the trained model
    model_file = sys.argv[2]

    # Call our main routine
    main(spark, data_file, model_file)
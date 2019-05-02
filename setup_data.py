#!/usr/bin/env python
# -*- coding: utf-8 -*-

# We need sys to get the command line arguments
import sys

from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline

def main(spark, data_file, new_data_file):
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
    indexer_item = StringIndexer(inputCol="track_id", outputCol="item", handleInvalid="skip")
    pipeline = Pipeline(stages=[indexer_user, indexer_item])
    transformed_train = pipeline.fit(train).transform(train)
    repartitioned_train =  transformed_train.repartition(5000, "item")
    repartitioned_train.write.parquet(new_data_file)
    
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('setup_data').getOrCreate()

    # Get the filename from the command line
    data_file = sys.argv[1]

    # And the location to store the trained model
    new_data_file = sys.argv[2]

    # Call our main routine
    main(spark, data_file, new_data_file)
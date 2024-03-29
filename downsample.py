#!/usr/bin/env python
# -*- coding: utf-8 -*-

# We need sys to get the command line arguments
import sys

from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.recommendation import ALS


def main(spark, train_data_file, val_data_file, test_data_file, model_file):
    '''
    Parameters
    ----------
    spark : SparkSession object
    data_file : string, path to the parquet file to load
    model_file : string, path to store the serialized model file
    '''

    # Load the parquet file
    train = spark.read.parquet(train_data_file)
    val = spark.read.parquet(val_data_file)
    test = spark.read.parquet(test_data_file)
    train1 = val.select("user_id").drop_duplicates()
    train2 = test.select("user_id").drop_duplicates()
    train2 = train1.union(train2)
    train_use = train.join(train2.select("user_id"), ["user_id"], "inner")
    train = train.join(train2.select("user_id"), ["user_id"], "leftanti")
    train = train.sample(withReplacement = False, fraction = 0.05)
    new_data = train.union(train_use)
    
    new_data.write.parquet(new_data_file)
    
    

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('downsample').getOrCreate()

    # Get the filename from the command line
    train_data_file = sys.argv[1]
    val_data_file = sys.argv[2]
    test_data_file = sys.argv[3]

    # And the location to store the trained model
    new_data_file = sys.argv[4]

    # Call our main routine
    main(spark, train_data_file, val_data_file, test_data_file, new_data_file)
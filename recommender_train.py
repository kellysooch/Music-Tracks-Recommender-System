#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 2: supervised model training
Usage:
    $ spark-submit supervised_train.py hdfs:/path/to/file.parquet hdfs:/path/to/save/model
'''


# We need sys to get the command line arguments
import sys

from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.recommendation import ALS


def main(spark, data_file, model_file):
    '''Main routine for supervised training
    Parameters
    ----------
    spark : SparkSession object
    data_file : string, path to the parquet file to load
    model_file : string, path to store the serialized model file
    '''

    # Load the parquet file
    df = spark.read.parquet(data_file)
    
    

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('supervised_train').getOrCreate()

    # Get the filename from the command line
    data_file = sys.argv[1]

    # And the location to store the trained model
    model_file = sys.argv[2]

    # Call our main routine
    main(spark, data_file, model_file)
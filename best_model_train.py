#!/usr/bin/env python
# -*- coding: utf-8 -*-

# We need sys to get the command line arguments
import sys

from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np

def main(spark, train_data_file, rank_val, reg, alpha_val,  model_file):
    '''
    Parameters
    ----------
    spark : SparkSession object
    data_file : string, path to the parquet file to load
    model_file : string, path to store the serialized model file
    '''

    # Load the parquet file
    train = spark.read.parquet(train_data_file)
    #val = spark.read.parquet(val_data_file)

    train = train.sample(withReplacement = False, fraction = .1)

    #transform data
    indexer_user = StringIndexer(inputCol="user_id", outputCol="user",
    handleInvalid="skip")
    indexer_item = StringIndexer(inputCol="track_id", outputCol="item",
    handleInvalid="skip")
    als = ALS(userCol = 'user', itemCol = 'item', implicitPrefs = True,
    ratingCol = 'count', rank = rank_val, regParam = reg, alpha = alpha_val)
    
    pipeline = Pipeline(stages = [indexer_user, indexer_item, als])
    model = pipeline.fit(train)
    model.save(model_file)

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('recommender_train').getOrCreate()

    # Get the filename from the command line
    train_data_file = sys.argv[1]
    rank_val = float(sys.argv[2])
    reg = float(sys.argv[3])
    alpha_val = float(sys.argv[4])

    # And the location to store the trained model
    model_file = sys.argv[5]

    # Call our main routine
    main(spark, train_data_file, rank_val, reg, alpha_val, model_file)

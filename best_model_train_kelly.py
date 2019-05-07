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

def main(spark, train_data_file, rank_val, reg, alpha_val, user_indexer_model, item_indexer_model, model_file):
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

    #transform data
    indexer_user = StringIndexer(inputCol="user_id", outputCol="user",
    handleInvalid="skip").fit(train)
    indexer_item = StringIndexer(inputCol="track_id", outputCol="item",
    handleInvalid="skip").fit(train)
    als = ALS(userCol = 'user', itemCol = 'item', implicitPrefs = True,
    ratingCol = 'count', rank = rank_val, regParam = reg, alpha = alpha_val)
    
    pipeline = Pipeline(stages = [indexer_user, indexer_item, als])
    train = indexer_user.transform(train)
    train = inexer_item.transform(train)
    model = als.fit(train)
    indexer_user.save(user_indexer_model)
    indexer_item.save(item_indexer_model)
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
    user_indexer_model = sys.argv[5]
    item_indexer_model = sys.argv[6]
    model_file = sys.argv[7]

    # Call our main routine
    main(spark, train_data_file, rank_val, reg, alpha_val, user_indexer_model, item_indexer_model, model_file)
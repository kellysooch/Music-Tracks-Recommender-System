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


def main(spark, train_data_file, val_data_file):
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
    
    indexer_user = StringIndexer(inputCol="user_id", outputCol="user",
    handleInvalid="keep").fit(train)
    indexer_item = StringIndexer(inputCol="track_id", outputCol="item",
    handleInvalid="keep").fit(train)
    
    rank = [1, 5, 10] #default is 10
    regularization = [ .01, .1, 1] #default is 1
    alpha = [ .01, .1, 1] #default is 1
    
    train = indexer_user.transform(train)
    train = indexer_item.transform(train)
    val = indexer_user.transform(val)
    val = indexer_item.transform(val)
    
    rank_list = []
    reg_list = []
    alpha_list = []
    rmses = []

    for i in rank:
        for j in regularization:
            for k in alpha:
                als = ALS(userCol = 'user', itemCol = 'item', implicitPrefs =
                          True, ratingCol = 'count', rank = i, regParam = j, alpha = k)
                output = als.fit(train)
                predictions = output.transform(val)
                evaluator = RegressionEvaluator(metricName="rmse", labelCol="count",predictionCol="prediction")
                rmse = evaluator.evaluate(predictions)
                rank_list.append(i)
                reg_list.append(j)
                alpha_list.append(k)
                rmses.append(rmse)
    
    print(rank_list)
    print(reg_list)
    print(alpha_list)
    print(rmses)
    print('Min RMSE value: %f' %min(rmses))
    ind = np.argmin(rmses)
    print('Rank: %f' %rank_list[ind])
    print('Reg: %f' %reg_list[ind])
    print('Alpha: %f' %alpha_list[ind])
    
    

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('recommender_train').getOrCreate()

    # Get the filename from the command line
    train_data_file = sys.argv[1]
    val_data_file = sys.argv[2]

    # Call our main routine
    main(spark, train_data_file, val_data_file)
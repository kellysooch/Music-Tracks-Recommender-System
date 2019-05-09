#!/usr/bin/env python
# -*- coding: utf-8 -*-

# We need sys to get the command line arguments
import sys

from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.feature import StringIndexerModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np


def main(spark, indexer_user, indexer_item, train_data_file, val_data_file):
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
    
    user_index = StringIndexerModel.load(indexer_user)
    item_index = StringIndexerModel.load(indexer_item)
    
    train = user_index.fit(train)
    train = item_index.fit(train)
    
    rank = [10, 20, 30] #default is 10
    regularization = [ .01, .1, 1] #default is 1
    alpha = [ .1, .5, 1] #default is 1
    
    train = indexer_user.transform(train)
    train = indexer_item.transform(train)
#     val = indexer_user.transform(val)
#     val = indexer_item.transform(val)
    
    rank_list = []
    reg_list = []
    alpha_list = []
    precisions = []

    for i in rank:
        for j in regularization:
            for k in alpha:
                als = ALS(userCol = 'user', itemCol = 'item', implicitPrefs = True, 
                          ratingCol = 'count', rank = i, regParam = j, alpha = k)
                model = als.fit(train)
                subset = val.select('user').distinct()
                predictions = model.predictForUserSubset(subset, 50)
                predictions = predictions.select("user", col("recommendations.item").alias("item")).sort('user')
                val = val.sort('user')
                predictionAndLabels = predictions.join(val,["user"], "inner").rdd.map(lambda tup: (tup[1], tup[2]))
                metrics = RankingMetrics(predictionAndLabels)
                precision = metrics.precisionAt(500)
                rank_list.append(i)
                reg_list.append(j)
                alpha_list.append(k)
                precisions.append(precision)
    
    print(rank_list)
    print(reg_list)
    print(alpha_list)
    print(rmses)
    print('Min precision value: %f' %min(precisions))
    ind = np.argmin(rmses)
    print('Rank: %f' %rank_list[ind])
    print('Reg: %f' %reg_list[ind])
    print('Alpha: %f' %alpha_list[ind])
    
    

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('recommender_train').getOrCreate()

    # Get the filename from the command line
    indexer_user = sys.argv[1]
    indexer_item = sys.argv[2]
    
    train_data_file = sys.argv[3]
    val_data_file = sys.argv[4]

    # Call our main routine
    main(spark, indexer_user, indexer_item, train_data_file, val_data_file)
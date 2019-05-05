#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 1: supervised model testing

Usage:

    $ spark-submit supervised_test.py hdfs:/path/to/load/model.parquet hdfs:/path/to/file

'''


# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics 
#from pyspark import SparkContext, SparkConf
#conf = SparkConf().setAppName(appName).setMaster(master)
#sc = SparkContext(conf=conf)

# TODO: you may need to add imports here


def main(spark, model_file, data_file):
    '''Main routine for supervised evaluation

    Parameters
    ----------
    spark : SparkSession object

    model_file : string, path to store the serialized model file

    data_file : string, path to the parquet file to load
    '''

    ###
    # TODO: YOUR CODE GOES HERE
    #import model
    pipeline = PipelineModel.load(model_file)
    
    #import test data
    test = spark.read.parquet(data_file)
   # test.createOrReplaceTempView('test')

    #model predictions
    predictions = pipeline.transform(test)
    #data = pipeline.transform(test)
    #userRecs = predictions.rdd.recommendForAllUsers(500)

    #grouped = predictions.sort(['prediction'], ascending =
    #False).groupby('user').collect(500)

    #evaluator = RegressionEvaluator(metricName="rmse", labelCol = 'count',
    #predictionCol = 'prediction')
    
    #rmse = evaluator.evaluate(userRecs)
    #print('RMSE = ' + str(rmse))
    
    predictionAndLabels = predictions.select('prediction',
    'count')
    metrics = RankingMetrics(predictionAndLabels)
    precision = metrics.precisionAt(500)
    ndcg = metrics.ndcgAt(500)

    print('Precision: %f' %precision)
    print('NDCG: %f' %ndcg)



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('supervised_test').getOrCreate()

    # And the location to store the trained model
    model_file = sys.argv[1]

    # Get the filename from the command line
    data_file = sys.argv[2]

    # Call our main routine
    main(spark, model_file, data_file)
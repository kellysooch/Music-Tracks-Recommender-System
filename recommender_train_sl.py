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
    train_sample = train.sample(withReplacement = False, fraction = .01)
    indexer_user = StringIndexer(inputCol="user_id", outputCol="user", handleInvalid="skip")
    indexer_item = StringIndexer(inputCol="track_id", outputCol="item", handleInvalid="skip")
    
    als = ALS(userCol="user", itemCol="item", implicitPrefs = True, ratingCol="count")
    
    #grid search values
    rank = [5, 10, 15] #default is 10
    regularization = [ .01, .1, 1, 10] #default is 1
    alpha = [ .01, .1, 1, 10] #default is 1

    #pipeline and crossvalidation
    pipeline = Pipeline(stages = [indexer_user, indexer_item, als])
    paramGrid = ParamGridBuilder().addGrid(als.rank, rank).addGrid(als.regParam,
    regularization).addGrid(als.alpha, alpha).build()

    crossval = CrossValidator(estimator = pipeline, estimatorParamMaps =
    paramGrid, evaluator = RegressionEvaluator(metricName="rmse",
    labelCol="count",predictionCol="prediction"))

    #create model
    model = crossval.fit(train_sample)
    modelbestModel.write().overwrite().save(model_file)


if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('recommender_train').getOrCreate()

    # Get the filename from the command line
    data_file = sys.argv[1]

    # And the location to store the trained model
    model_file = sys.argv[2]

    # Call our main routine
    main(spark, data_file, model_file)

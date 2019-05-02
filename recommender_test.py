#!/usr/bin/env python
# -*- coding: utf-8 -*-

# We need sys to get the command line arguments
import sys

from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.recommendation import ALS
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

def main(spark, model_file, test_file):
    '''
    Parameters
    ----------
    spark : SparkSession object
    data_file : string, path to the parquet file to load
    model_file : string, path to store the serialized model file
    '''

    # Load the parquet file
    test = spark.read.parquet(test_file)
    indexer_user = StringIndexer(inputCol="user_id", outputCol="user", handleInvalid="skip")
    indexer_item = StringIndexer(inputCol="track_id", outputCol="item", handleInvalid="skip")
    
    pipeline = Pipeline(stages=[indexer_user, indexer_item])
    transformed_train = pipeline.fit(test).transform(test)
    
    model = PipelineModel.load(model_file)
    
    predictions = model.transform(test)
    user_recs = model.recommendForAllUsers(500)
#     sorted_preds = predictions.sort('prediction', ascending = False)
#     sorted_preds.createOrReplaceTempView('df')
#     pred = spark.sql("SELECT user, max(item) as item, max(prediction) as prediction, max(count) as count, COUNT(*) as num FROM df GROUP BY user HAVING num <= 500")
    evaluator = RegressionEvaluator(metricName="rmse",labelCol="count",predictionCol="prediction")
    rmse = evaluator.evaluate(user_recs)
    print("Root-mean-square error = " + str(rmse))

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('recommender_train').getOrCreate()

    # Get the model from the command line
    model_file = sys.argv[1]

    # And the test file
    test_file = sys.argv[2]

    # Call our main routine
    main(spark, model_file, test_file)
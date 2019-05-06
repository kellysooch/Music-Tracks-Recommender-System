#!/usr/bin/env python
# -*- coding: utf-8 -*-

# We need sys to get the command line arguments
import sys

from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.recommendation import ALS
from pyspark.ml.recommendation import ALSModel
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
from pyspark import SparkContext
from pyspark import SparkConf

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
    
    model = PipelineModel.load(model_file)
    
    test_transformed = model.transform(test)
    predictionAndLabels = test_transformed.select(["prediction", "count"]).rdd.map(lambda row: (row.prediction, row.count)).take(10)
    print(predictionAndLabels)
#     rdd = sc.parallelize([[1.0], [0.0]])
#     metrics = RankingMetrics(rdd)
#     precision = metrics.precisionAt(1)
#     ndcg = metrics.ndcgAt(500)

#     print('Precision: %f' %precision)
#     print('NDCG: %f' %ndcg)


if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('recommender_train').getOrCreate()

    # Get the model from the command line
    model_file = sys.argv[1]

    # And the test file
    test_file = sys.argv[2]

    # Call our main routine
    main(spark, model_file, test_file)
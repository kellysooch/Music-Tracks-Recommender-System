#!/usr/bin/env python
# -*- coding: utf-8 -*-

# We need sys to get the command line arguments
import sys

from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.recommendation import ALSModel
from pyspark.ml import PipelineModel
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.feature import StringIndexerModel
from pyspark.sql.functions import col
import pyspark.sql.functions as F

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
    test = test.sort('user')
    test.createOrReplaceTempView('test_table')
    test = spark.sql('SELECT * FROM test_table LIMIT 1000')
#     print(test.take(10))
#     test = test.sample(withReplacement = False, fraction = 0.4)
    model = ALSModel.load(model_file)
    
    user_subset = test.select("user").distinct()
    user_subset = model.recommendForUserSubset(user_subset, 500)
    
    user_subset = user_subset.select("user", col("recommendations.item").alias("item"))
    user_subset = user_subset.sort('user')
    print(user_subset.take(10))
    print("sort user")
    predictionAndLabels = user_subset.join(test,["user"], "inner").rdd.map(lambda tup: (tup[1], tup[2]))
#     print(predictionAndLabels.take(10))
    print("joined predictions and counts")

#     metrics = RankingMetrics(predictionAndLabels)
    print("made metrics")
    MAP = metrics.meanAveragePrecision
    precision = metrics.precisionAt(500)
    ndcg = metrics.ndcgAt(500)
    
    print('MAP: %f' %MAP)
    print('Precision: %f' %precision)
    print('NDCG: %f' %ndcg)


if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('recommender_test').getOrCreate()

    # Get the model from the command line
    model_file = sys.argv[1]

    # And the test file
    test_file = sys.argv[2]

    # Call our main routine
    main(spark, model_file, test_file)
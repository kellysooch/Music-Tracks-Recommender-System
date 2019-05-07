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
from pyspark.sql.functions import col

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
    print("read file")
    
    model = PipelineModel.load(model_file)
    print("loaded model")
    
    test_transformed = model.transform(test)
    print("transformed test file")
    predictions = test_transformed.select(["user","item","prediction"]).rdd.map(lambda r: ((r.user, r.item), r.prediction))
    print("made predictions tuple")
    test_select = test_transformed.select(col("user"), col("item"), col("count").alias("prediction"))
    print("renamed cols")
    ratingsTuple = test_select.select(["user","item","prediction"]).rdd.map(lambda r: ((r.user, r.item), r[2]))
    print("made label tuple")
    predictionAndLabels = predictions.join(ratingsTuple).map(lambda tup: tup[1])
    print("joined predictions and counts")

    metrics = RankingMetrics(predictionAndLabels)
    print("made metrics")
    precision = metrics.precisionAt(500)
    print("made precision")
    ndcg = metrics.ndcgAt(500)
    print("made ndcg")

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
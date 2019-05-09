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

def main(spark, user_indexer_model, item_indexer_model, model_file, test_file):
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
    test = test.sort('user', ascending=False)
    print("sort test")
    test.createOrReplaceTempView('test_table')
    test = spark.sql('SELECT * FROM test_table LIMIT 50000')
#     test = test.sample(withReplacement = False, fraction = 0.4)    
    print("sql test")
    model = ALSModel.load(model_file)
    print("loaded model")
    
    user_subset = test.select("user").distinct()
#     user_subset = user_subset.sample(withReplacement = False, fraction = 0.5)
    print("select users")
    user_subset = model.recommendForUserSubset(user_subset, 500)
    
    user_subset = user_subset.select("user", col("recommendations.item").alias("item"))
    print("select recs")
    user_subset = user_subset.sort('user', ascending=False)
#     relevant_docs = test.groupBy('user').agg(F.collect_list('item').alias('item')).select("user", "item")
    print("sort user")
#     user_subset = user_subset.repartition(500)
#     test = test.repartition(500)
    predictionAndLabels = user_subset.join(test,["user"], "inner").rdd.map(lambda tup: (tup[1], tup[2]))
    print("joined predictions and counts")

    metrics = RankingMetrics(predictionAndLabels)
    print("made metrics")
    precision = metrics.precisionAt(500)
    ndcg = metrics.ndcgAt(500)

    print('Precision: %f' %precision)
    print('NDCG: %f' %ndcg)


if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('recommender_test').getOrCreate()

    # Get the model from the command line
    user_indexer_model = sys.argv[1]
    item_indexer_model = sys.argv[2]
    model_file = sys.argv[3]

    # And the test file
    test_file = sys.argv[4]

    # Call our main routine
    main(spark, user_indexer_model, item_indexer_model, model_file, test_file)
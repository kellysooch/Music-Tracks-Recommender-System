#!/usr/bin/env python
# -*- coding: utf-8 -*-

# We need sys to get the command line arguments
import sys

from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.ml import PipelineModel
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.feature import StringIndexerModel
from pyspark.sql.functions import col

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
    test = test.sample(withReplacement = False, fraction = 0.001)
    print("sample file")
    user_index = StringIndexerModel.load(user_indexer_model)
    item_index = StringIndexerModel.load(item_indexer_model)
    test = user_index.transform(test)
    test = item_index.transform(test)
    print("transform file")
    model = ALSModel.load(model_file)
    print("loaded model")
    
#     test_transformed = model.transform(test)
    print("transformed test file")
#     print(test_transformed.take(10))
    user_recs = model.recommendForAllUsers(5)
    print(user_recs.head(5))
    top5 = user_recs.where(user_recs.user == 17918.0).select("recommendations.item", "recommendations.rating").collect()
    print("collect")
    print(top5)
#     predictions = top_predictions.select(col("user"), col("recommendations.item").alias("item")).rdd.map(lambda r: (r.user, [r.item]))
#     top_predictions.createOrReplaceTempView('mytable')
#     results = spark.sql('SELECT * from mytable LIMIT 5')
#     print("top 5")
#     print(results.show())
    
    print("made predictions tuple")
    relevant_docs = test.select(["user","item"]).rdd.map(lambda r: (r.user, [r.item])).reduceByKey(lambda p, q: p+q)
    print(relevant_docs.take(10))
#     test_select = test_transformed.select(col("user"), col("item"), col("count").alias("prediction"))
#     ratingsTuple = test_select.rdd.map(lambda r: (r.user, [r.prediction])).reduceByKey(lambda p, q: p+q)
#     print("made label tuple")
#     predictionAndLabels = predictions.join(ratingsTuple).map(lambda tup: tup[1])
#     predictionAndLabels = predictions.join(relevant_docs).map(lambda tup: tup[1])
#     print(predictionAndLabels.take(10))
#     print("joined predictions and counts")

#     metrics = RankingMetrics(predictionAndLabels)
#     print("made metrics")
#     precision = metrics.precisionAt(500)
#     print("made precision")
#     ndcg = metrics.ndcgAt(500)
#     print("made ndcg")

#     print('Precision: %f' %precision)
#     print('NDCG: %f' %ndcg)


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
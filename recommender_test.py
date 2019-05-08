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
#     test = test.sample(withReplacement = False, fraction = 0.001)
#     print("sample file")
    user_index = StringIndexerModel.load(user_indexer_model)
    item_index = StringIndexerModel.load(item_indexer_model)
    test = user_index.transform(test)
    test = item_index.transform(test)
    print("transform file")
#     model = ALSModel.load(model_file)
#     print("loaded model")
    
# #     test_transformed = model.transform(test)
#     print("transformed test file")
# #     print(test_transformed.take(10))
#     user_subset = test.select("user").distinct()
#     print("select users")
#     user_recs = model.recommendForUserSubset(user_subset, 500)
    
#     top5 = user_recs.select("user", col("recommendations.item").alias("item")).rdd.map(lambda r: (r.user, r.item))
#     top5 = top5.repartition(2000)
#     print("select rdd")
    
#     print("made predictions tuple")
#     test.createOrReplaceTempView("mytable") 
#     test_group = spark.sql("SELECT user, COUNT(*) as counting FROM mytable GROUP BY user ORDER BY counting LIMIT 50")
#     test_group.show()
    relevant_docs = test.select(["user","item"]).rdd.map(lambda r: (r.user, [r.item])).reduceByKey(lambda p, q: p+q)
    relevant_docs.filter(relevant_docs.user == 207714).show()
#     test_select = test_transformed.select(col("user"), col("item"), col("count").alias("prediction"))
#     ratingsTuple = test_select.rdd.map(lambda r: (r.user, [r.prediction])).reduceByKey(lambda p, q: p+q)
    print("made label tuple")
#     predictionAndLabels = predictions.join(ratingsTuple).map(lambda tup: tup[1])
    predictionAndLabels = top5.join(relevant_docs).map(lambda tup: tup[1])
#     print(predictionAndLabels.take(10))
    predictionAndLabels = predictionAndLabels.repartition(1000)
    print("joined predictions and counts")

    metrics = RankingMetrics(predictionAndLabels)
    print("made metrics")
    precision = metrics.precisionAt(500)
#     print("made precision")
    ndcg = metrics.ndcgAt(500)
#     print("made ndcg")

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
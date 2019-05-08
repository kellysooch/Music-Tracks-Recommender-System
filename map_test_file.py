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

def main(spark, user_indexer_model, item_indexer_model, test_file, save_test):
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
    user_index = StringIndexerModel.load(user_indexer_model)
    item_index = StringIndexerModel.load(item_indexer_model)
    test = user_index.transform(test)
    test = item_index.transform(test)
    
    
#     relevant_docs = test.select(["user","item"]).rdd.map(lambda r: (r.user, [r.item])).reduceByKey(lambda p, q: p+q)
    relevant_docs = test.groupBy('user').agg(F.collect_list('item'))
    relevant_docs = relevant_docs.repartition(1000)
#     relevant_docs = relevant_docs.toDF()
    relevant_docs.write.parquet(save_test)

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('map_test_file').getOrCreate()

    # Get the model from the command line
    user_indexer_model = sys.argv[1]
    item_indexer_model = sys.argv[2]
    test_file = sys.argv[3]

    # And the test file
    save_test = sys.argv[4]

    # Call our main routine
    main(spark, user_indexer_model, item_indexer_model, test_file, save_test)
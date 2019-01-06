import findspark
findspark.init()

# create spark context
print('Creates & returns Spark Context and SQL Context \n Usage: spark, sqlContext = createSpark()')

def createSpark():
    # create spark session
    from pyspark.sql import SparkSession
    spark = SparkSession \
        .builder \
        .appName("PySpark") \
        .getOrCreate()

    # create SQL context
    from pyspark.sql import SQLContext
    sqlContext = SQLContext(spark)
    
    return spark,sqlContext


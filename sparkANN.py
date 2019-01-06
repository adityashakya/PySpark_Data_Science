# Finding Spark Installation on the system
# %SPARK_HOME% must be set in the environment variables for this to work
import findspark
findspark.init()

# create spark context
from pyspark import SparkContext, SparkConf
conf = SparkConf()
sc = SparkContext(conf=conf)

# create spark session
from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("PySpark ANN") \
    .getOrCreate()

# create SQL context
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)


# read the csv file
data = sqlContext.read.format("com.databricks.spark.csv")\
.option("header", "true")\
.option("inferschema", "true")\
.option("mode", "DROPMALFORMED")\
.load("data/Churn_Modelling.csv")

#  required data modification on the "Exited" column in "data"
import pyspark.sql.functions as F

# converting integer values in Exited column to string, so that we can train the stringIndexer to get output labels for the model.
data =data.withColumn("Exited", F.col("Exited").cast('boolean').cast('string'))

# splitting the training and test data
(trainingData, testData) = data.randomSplit([0.8, 0.2])

# setting up the stages of the pipeline
# imports
from pyspark.ml.feature import StringIndexer, IndexToString, VectorAssembler, OneHotEncoderEstimator, MinMaxScaler
from pyspark.ml.classification import MultilayerPerceptronClassifier

# stages
stages=[] # to hold all stages in sequence

# stage for output label encoding
stages.append(StringIndexer(inputCol='Exited',  outputCol='s_exited', handleInvalid='skip').fit(trainingData))

# string index geography
stages.append(StringIndexer(inputCol ='Geography', outputCol='s_geography', handleInvalid='keep'))

# One hot encode geography
stages.append(OneHotEncoderEstimator(inputCols=['s_geography'], outputCols=['oh_s_geography']))

# String index gender
stages.append(StringIndexer(inputCol = 'Gender', outputCol='s_gender', handleInvalid='keep'))

# one hot encoding gender
stages.append(OneHotEncoderEstimator(inputCols=['s_gender'], outputCols=['oh_s_gender']))

# stage for feature/vector assembler
stages.append(VectorAssembler(inputCols=['oh_s_gender','oh_s_geography','CreditScore',
 'Age',
 'Tenure',
 'Balance',
 'NumOfProducts',
 'HasCrCard',
 'IsActiveMember',
 'EstimatedSalary'], outputCol='features'))

# stage for scaling the features using MinMax scaler
stages.append(MinMaxScaler(inputCol='features', outputCol='scaledfeatures'))

# stage for MultilayerPerceptronClassifier(ANN implementation in Spark)
layers = [13,6,6,6,2] # 13- input features, two hidden layers with 5 neurons, output layer with 2 neurons(for 2 o/p labels)
stages.append(MultilayerPerceptronClassifier(labelCol="s_exited", featuresCol="scaledfeatures",
                                         maxIter=200, layers=layers))

#stage for reverse indexing the prediction label
stages.append(IndexToString(inputCol='prediction', outputCol='lab_prediction', labels=stages[0].labels))


#  making the pipeline model
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=stages) # Making Pipeline


# making/Training the model using trainingData
model = pipeline.fit(trainingData)

# making the predictions on 
predictions = model.transform(testData)

# evaluating the model performance
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol='s_exited',predictionCol='prediction', metricName='accuracy')
accuracy = evaluator.evaluate(predictions)
print("Accuracy  = %g and Test Error = %g"%(accuracy, 1-accuracy))

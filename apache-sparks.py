from warnings import filterwarnings
filterwarnings('ignore')
import pyspark
from pyspark import SparkContext
import findspark
findspark.init("sparksın bulunduğu dosyanın yolu")
import pyspark
from pyspark import SparkContext
from pyspark import SparkContext
import pyspark
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark import SparkContext
import seaborn as sns


sc = SparkContext(master = "local")
sc.version
sc.sparkUser()
sc.appName
dir(sc)
sc.stop()



spark = SparkSession.builder \
    .master("local") \
    .appName("pyspark_uygulama") \
    .config("spark.executer.memory", "16gb") \
    .getOrCreate()



sc = spark.sparkContext
sc

# Temel DataFrame İşlemleri

spark_df = spark.read.csv("diabetes.csv", header = True, inferSchema = True)
spark_df.printSchema()
type(spark_df)
spark_df.cache()
type(spark_df)



df = sns.load_dataset("diamonds")
df = df.select_dtypes(include = ["float64","int64"])
type(df)
df.head()
spark_df.head()
df.dtypes
spark_df.dtypes
df.ndim
spark_df.size
spark_df.show(2)
spark_df.count()
len(spark_df.columns)
spark_df.describe().show()

#degisken secme
spark_df.describe("Glucose").show()
spark_df.select("Glucose","Pregnancies").show(5)
spark_df.select("Glucose").distinct().count()
spark_df.select("Glucose").dropDuplicates().count()
spark_df.crosstab("Outcome","Pregnancies").show()
spark_df.dropna().show(3)

#gozlem secme
spark_df.filter(spark_df.Age >40).count()
spark_df.groupby("Outcome").count().show()
spark_df.groupby("Outcome").agg({"BMI": "mean"}).show()
spark_df.withColumn("yeni_degisken", spark_df.BMI / 2).select("BMI","yeni_degisken").show(5)
spark_df.withColumnRenamed("Outcome","bagimli_degisken").columns
spark_df.show(3)
spark_df.drop("Insulin").columns
a = spark_df.groupby("Outcome").count().toPandas()
a.iloc[1,1]

#SQL İşlemleri
spark_df.registerTempTable("table_df")
spark.sql("show databases").show()
spark.sql("show tables").show()
spark.sql("select Glucose from table_df").show(5)
spark.sql("select Outcome, mean(Glucose) from table_df group by Outcome").show(5)

# Büyük Veri Görselleştirme

import matplotlib.pyplot as plt
import seaborn as sns

sns.barplot(x = "Outcome", y = spark_df.Outcome.index, data = spark_df)
sdf = spark_df.toPandas()
sdf.head()
sns.barplot(x = "Outcome", y = sdf.Outcome.index, data = sdf)

# Uçtan Büyük Veride Makine Öğrenmesi
sc.stop()
import pyspark
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark import SparkContext

spark = SparkSession.builder \
    .master("local") \
    .appName("churn_modellemesi") \
    .config("spark.executer.memory", "16gb") \
    .getOrCreate()



sc = spark.sparkContext
sc
spark_df = spark.read.csv("churn.csv",header = True,inferSchema = True,sep = ",")
spark_df.cache()


spark_df.printSchema()
spark_df.show(5)
spark_df = spark_df.toDF(*[c.lower() for c in spark_df.columns])
spark_df.show(5)
#df.columns = map(str.lower, df.columns)

spark_df = spark_df.withColumnRenamed("_c0", "index")
spark_df.show(2)
spark_df.count()


len(spark_df.columns)
spark_df.columns
spark_df.distinct().count()
spark_df.select("names").distinct().count()
spark_df.groupby("names").count().sort("count", ascending = False).show(3)
spark_df.filter(spark_df.names == "Jennifer Wood").show()
spark_df.select("names").dropDuplicates().groupBy("names").count().sort("count",ascending = False).show(3)
spark_df.where(spark_df.index == 439).select("names").show()

jen = spark_df.where(spark_df.index == 439).collect()[0]["names"]
type(jen)
dir(jen)
jen.upper()

## Keşifçi Veri Analizi
print(spark_df.describe().show())
spark_df.select("age","total_purchase", "account_manager", "years","num_sites","churn").describe().toPandas().transpose()
spark_df.filter(spark_df.age > 47).count()
spark_df.groupby("churn").count().show()
spark_df.groupby("churn").agg({"total_purchase": "mean"}).show()
spark_df.groupby("churn").agg({"years": "mean"}).show()
kor_data = spark_df.drop("index","names").toPandas()
import seaborn as sns
sns.pairplot(kor_data, hue = "churn")
sns.pairplot(kor_data, vars = ["age", "total_purchase","years","num_sites"],hue = "churn",kind = "reg")


#Veri ön işleme
spark_df = spark_df.dropna()
spark_df = spark_df.withColumn("age_kare", spark_df.age**2)
spark_df.show(3)
from pyspark.ml.feature import StringIndexer
stringIndexer = StringIndexer(inputCol = "churn", outputCol = "label")
indexed = stringIndexer.fit(spark_df).transform(spark_df)
indexed.dtypes
spark_df = indexed.withColumn("label", indexed["label"].cast("integer"))
spark_df.dtypes


#bagimsiz degiskenlerin ayarlanmasi
from pyspark.ml.feature import VectorAssembler
spark_df.columns
bag = ["age","total_purchase", "account_manager","years","num_sites"]
vectorAssembler = VectorAssembler(inputCols = bag, outputCol = "features")
va_df = vectorAssembler.transform(spark_df)
final_df = va_df.select(["features","label"])
final_df.show()


## Test-train
splits = final_df.randomSplit([0.7,0.3])
train_df = splits[0]
test_df = splits[1]
train_df
test_df

# GBM ile Müşteri Terk Modellemesi
from pyspark.ml.classification import GBTClassifier
gbm = GBTClassifier(maxIter = 10, featuresCol = "features", labelCol = "label")
gbm_model = gbm.fit(train_df)
y_pred = gbm_model.transform(test_df)
y_pred
ac = y_pred.select("label","prediction")
ac.filter(ac.label == ac.prediction).count() / ac.count()

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

evaluator = BinaryClassificationEvaluator()
paramGrid = (ParamGridBuilder().addGrid(gbm.maxDepth, [2, 4, 6]).addGrid(gbm.maxBins, [20, 30]).addGrid(gbm.maxIter, [10, 20]).build())
cv = CrossValidator(estimator= gbm, estimatorParamMaps = paramGrid, evaluator=evaluator, numFolds= 10)
cv_model = cv.fit(train_df)
y_pred = cv_model.transform(test_df)
ac = y_pred.select("label","prediction")
ac.filter(ac.label == ac.prediction).count() / ac.count()
evaluator.evaluate(y_pred)

























from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, explode
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd, os

RATINGS_PARQUET = "work/ratings_parquet"
PRODUCTS_CSV = "work/products"
OUT_DIR = "work/out" 

TOP_N = 5
ALS_RANK = 15
ALS_MAXITER = 12
ALS_REG = 0.05

spark = (
    SparkSession.builder
    .appName("EcomALSImplicit")
    .config("spark.driver.memory", "4g")       
    .config("spark.executor.memory", "4g")
    .config("spark.sql.shuffle.partitions", "50") 
    .getOrCreate()
)

df = spark.read.parquet(RATINGS_PARQUET) \
               .select("user_id", "product_id", "rating") \
               .dropna()

df = df.select(
    col("user_id").cast("int").alias("user_id"),
    col("product_id").cast("int").alias("product_id"),
    col("rating").cast("float").alias("rating")
)

train, test = df.randomSplit([0.8, 0.2], seed=42)

als = ALS(
    userCol="user_id",
    itemCol="product_id",
    ratingCol="rating",
    implicitPrefs=True,
    coldStartStrategy="drop",
    rank=ALS_RANK,
    maxIter=ALS_MAXITER,
    regParam=ALS_REG
)
model = als.fit(train)

pred = model.transform(test)
evaluator = RegressionEvaluator(labelCol="rating", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(pred)
print(f"✅ RMSE (implicit proxy): {rmse:.4f}")

user_recs = model.recommendForAllUsers(TOP_N)
flat_recs = (
    user_recs
    .select(col("user_id"), explode("recommendations").alias("rec"))
    .select(
        col("user_id"),
        col("rec.product_id").alias("product_id"),
        col("rec.rating").alias("pred_score")
    )
)

products = spark.read.option("header", True).csv(PRODUCTS_CSV)
joined_recs = (
    flat_recs.join(products, on="product_id", how="left")
)

joined_recs.write.mode("overwrite").option("header", True).csv(f"{OUT_DIR}/top_recommendations_pretty")

summary = (
    df.groupBy("product_id")
      .agg(
          count("*").alias("interaction_count"),
          avg("rating").alias("avg_feedback")
      )
      .orderBy(col("interaction_count").desc())
)
summary.write.mode("overwrite").option("header", True).csv(f"{OUT_DIR}/analytics_summary")

os.makedirs(OUT_DIR, exist_ok=True)
pd.DataFrame([{"metric": "rmse", "value": rmse}]).to_csv(f"{OUT_DIR}/model_metrics.csv", index=False)

print(f"✅ Outputs written to: {OUT_DIR}")
spark.stop()

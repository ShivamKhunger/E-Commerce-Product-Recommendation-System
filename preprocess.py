
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit
from pyspark.sql.types import IntegerType, FloatType, StringType

INPUT_CSV = "2019-Nov-small.csv"          
OUTPUT_PARQUET = "work/ratings_parquet"  
OUTPUT_PRODUCTS = "work/products"         

spark = (
    SparkSession.builder
    .appName("EcomPreprocessImplicit")
    .config("spark.driver.memory", "4g")        
    .config("spark.executor.memory", "4g")
    .config("spark.sql.shuffle.partitions", "50") 
    .getOrCreate()
)

df = spark.read.csv(INPUT_CSV, header=True, inferSchema=True).select(
    "user_id", "product_id", "event_type", "category_code", "brand", "price"
)

df = df.dropna(subset=["user_id", "product_id", "event_type"]) \
       .withColumn("user_id", col("user_id").cast(IntegerType())) \
       .withColumn("product_id", col("product_id").cast(IntegerType())) \
       .withColumn("event_type", col("event_type").cast(StringType()))

df = df.withColumn(
    "rating",
    when(col("event_type") == "purchase", lit(5.0))
    .when(col("event_type") == "cart", lit(3.0))
    .when(col("event_type") == "view", lit(1.0))
    .otherwise(lit(0.0))
)

ratings = df.select("user_id", "product_id", "rating")

products = (
    df.select("product_id", "category_code", "brand", "price")
      .dropna(subset=["product_id"])
      .dropDuplicates(["product_id"])
)
products.write.mode("overwrite").option("header", True).csv(OUTPUT_PRODUCTS)

ratings.write.mode("overwrite").parquet(OUTPUT_PARQUET)

count_rows = ratings.count()
print(f"✅ Prepared implicit ratings rows: {count_rows}")

spark.stop()

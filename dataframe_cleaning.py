from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, lower, trim, when
from pyspark.sql.types import StringType

# Initialize Spark session
spark = SparkSession.builder \
    .appName("BigDataFrameCleaning") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .getOrCreate()

# Example: Load a large dataset from a CSV file
df = spark.read.csv("path/to/large_dataset.csv", header=True, inferSchema=True)

# 1. Drop Duplicates
df = df.dropDuplicates()

# 2. Handle Missing Values
# a) Drop rows with missing values in specific columns
df = df.dropna(subset=["important_column1", "important_column2"])

# b) Fill missing values with a default value (e.g., "Unknown" for strings, 0 for numbers)
df = df.fillna({
    "column1": "Unknown",
    "column2": 0,
    "column3": "N/A"
})

# 3. Clean Column Data
# a) Remove special characters from a column and convert to lowercase
df = df.withColumn("cleaned_text", regexp_replace(col("text_column"), "[^a-zA-Z0-9\\s]", ""))
df = df.withColumn("cleaned_text", lower(col("cleaned_text")))

# b) Trim leading and trailing whitespace
df = df.withColumn("cleaned_text", trim(col("cleaned_text")))

# 4. Handle Categorical Columns
# a) Standardize categorical values (e.g., replace "yes", "YES", "Yes" with "yes")
df = df.withColumn("categorical_column", lower(trim(col("categorical_column"))))
df = df.withColumn("categorical_column", 
                   when(col("categorical_column") == "yes", "yes")
                   .when(col("categorical_column") == "no", "no")
                   .otherwise(col("categorical_column")))

# 5. Handle Outliers (Example: Cap values in a numeric column)
quantiles = df.approxQuantile("numeric_column", [0.05, 0.95], 0.0)
lower_bound = quantiles[0]
upper_bound = quantiles[1]

df = df.withColumn("numeric_column_capped", 
                   when(col("numeric_column") < lower_bound, lower_bound)
                   .when(col("numeric_column") > upper_bound, upper_bound)
                   .otherwise(col("numeric_column")))

# 6. Type Casting (Convert columns to appropriate data types)
df = df.withColumn("numeric_column", col("numeric_column").cast("double"))
df = df.withColumn("date_column", col("date_column").cast("date"))

# 7. Rename Columns (if needed)
df = df.withColumnRenamed("old_column_name", "new_column_name")

# 8. Drop Unnecessary Columns
df = df.drop("unnecessary_column1", "unnecessary_column2")

# 9. Repartition DataFrame to optimize processing
df = df.repartition(10)  # Repartition based on your cluster size

# Show a sample of cleaned data
df.show(10)

# Save the cleaned DataFrame to a new file or database
df.write.csv("path/to/cleaned_dataset.csv", header=True)

# Stop the Spark session
spark.stop()

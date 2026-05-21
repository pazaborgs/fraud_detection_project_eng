import yaml
from pyspark.sql import functions as F

# ============================================
# LOAD_CONFIG
# ============================================

def load_config(config_path="../config/config.yaml"):
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

# ============================================
# INGEST_BRONZE
# ============================================

def ingest_bronze(spark, config):
    """
    Reads raw data and saves it to the Bronze layer as a Delta table.
    """
    raw_path = config['paths']['raw'] 
    bronze_path = config['paths']['bronze']
    
    if not spark.catalog.tableExists(raw_path):
        raise Exception(f"Source table '{raw_path}' does not exist. Cannot proceed.")

    print(f"Reading raw data from {raw_path}...")
    df_raw = spark.read.table(raw_path)

    df_bronze = df_raw.withColumn("ingestion_timestamp", F.current_timestamp())

    print(f"Saving Bronze data to {bronze_path}...")
    df_bronze.write \
        .format("delta") \
        .mode("overwrite") \
        .saveAsTable(bronze_path)
    
    print("Bronze ingestion complete!")
    return df_bronze

# ============================================
# PROCESS_SILVER
# ============================================

def process_silver(spark, config):
    """
    Reads from Bronze, applies data typing/renaming, filters relevant types, 
    and saves to Silver partitioned by step.
    """

    bronze_path = config['paths']['bronze']
    silver_path = config['paths']['silver']
    fraud_types = config['business_rules']['fraud_types'] # ["TRANSFER", "CASH_OUT"]
    
    print(f"Reading Bronze data from {bronze_path}...")
    df_bronze = spark.read.table(bronze_path)
    
    df_silver = df_bronze.select(
        F.col("step").cast("int"),
        F.col("type").cast("string"),
        F.col("amount").cast("double"),
        F.col("nameOrig").cast("string").alias("name_orig"),
        F.col("oldbalanceOrg").cast("double").alias("old_balance_org"),
        F.col("newbalanceOrig").cast("double").alias("new_balance_orig"),
        F.col("nameDest").cast("string").alias("name_dest"),
        F.col("oldbalanceDest").cast("double").alias("old_balance_dest"),
        F.col("newbalanceDest").cast("double").alias("new_balance_dest"),
        F.col("isFraud").cast("int").alias("is_fraud"),
        F.col("isFlaggedFraud").cast("int").alias("is_flagged_fraud")
    )
    
    # Drop Duplicates
    df_silver = df_silver.dropDuplicates()

    # Filter only TRANSFER and CASH_OUT
    df_silver_filtered = df_silver.filter(F.col("type").isin(fraud_types))
    
    # Save as Delta, partitioned by step
    print(f"Saving Silver data to {silver_path} partitioned by step...")
    df_silver_filtered.write \
        .format("delta") \
        .mode("overwrite") \
        .partitionBy("step") \
        .saveAsTable(silver_path)
        
    print("Silver processing complete!")
    return df_silver_filtered

# ============================================
# PROCESS_GOLD
# ============================================

def process_gold(spark, config):
    """
    Reads Silver data, engineers features for ML (errors, ratios, time),
    and saves the final feature table to the Gold layer.
    """

    silver_path = config['paths']['silver']
    gold_path = config['paths']['gold']
    
    print(f"Reading Silver data from {silver_path}...")
    df_silver = spark.read.table(silver_path)

    df_transformed = df_silver.withColumn(
        "error_orig", 
        F.col("old_balance_org") - F.col("amount") - F.col("new_balance_orig")
    ).withColumn(
        "error_dest", 
        F.when(
            F.col("name_dest").startswith("M"), F.lit(0.0) # Merchants have no balance data
        ).otherwise(
            F.col("old_balance_dest") + F.col("amount") - F.col("new_balance_dest")
        )
    ).withColumn(
        "hour_of_day", 
        F.col("step") % 24
    ).withColumn(
        "ratio_amount_balance", 
        F.when(
            F.col("old_balance_org") > 0, F.col("amount") / F.col("old_balance_org")
        ).otherwise(F.lit(-1.0)) # Protection against division by zero
    ).withColumn(
        "type_idx", 
        F.when(F.col("type") == "TRANSFER", F.lit(0))
         .otherwise(F.lit(1)) # 1 = CASH_OUT
    ).withColumn(
        "dest_is_empty", 
        F.when(F.col("old_balance_dest") == 0, F.lit(1)) # 1 = Standard Mule Pattern
         .otherwise(F.lit(0))                            # 0 = Account with History
    )

    features_list = [
        "step", "hour_of_day", "type_idx", "amount",
        "old_balance_org", "error_orig", "old_balance_dest",
        "error_dest", "dest_is_empty", "ratio_amount_balance", "is_fraud"
    ]

    df_ml_final = df_transformed.select(features_list)
    
    # Save the final table to the Gold layer
    print(f"Saving Gold data to {gold_path}...")
    df_ml_final.write \
        .format("delta") \
        .mode("overwrite") \
        .saveAsTable(gold_path)
        
    print("Gold processing complete!")
    return df_ml_final


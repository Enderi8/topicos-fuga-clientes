#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import col, when, trim, regexp_replace

def parse_arguments():
    parser = argparse.ArgumentParser(description='Proceso de carga - Capa Curated Telco')
    parser.add_argument('--env', type=str, default='topicosa', help='Entorno: topicosa')
    parser.add_argument('--username', type=str, default='hadoop', help='Usuario HDFS')
    parser.add_argument('--base_path', type=str, default='/user', help='Ruta base en HDFS')
    parser.add_argument('--source_db', type=str, default='landing', help='Base de datos origen')
    parser.add_argument('--enable-validation', action='store_true', default=True, help='Activar validaciones')
    return parser.parse_args()

def create_spark_session(app_name="ProcesoCurated-TelcoChurn"):
    return SparkSession.builder \
        .appName(app_name) \
        .enableHiveSupport() \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
        .config("hive.exec.dynamic.partition", "true") \
        .config("hive.exec.dynamic.partition.mode", "nonstrict") \
        .getOrCreate()

def crear_database(spark, env, username, base_path):
    db_name = f"{env}_curated".lower()
    # Ubicación según tu estructura: /user/hadoop/datalake/TOPICOSA_CURATED
    db_location = f"{base_path}/{username}/datalake/{db_name.upper()}"
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name} LOCATION '{db_location}'")
    print(f"✅ Database '{db_name}' lista en: {db_location}")
    return db_name

# --- REGLAS DE CALIDAD PARA TU TABLA CUSTOMERS ---
def aplicar_reglas_calidad_customers(df, enable_validation=True):
    """
    Transformaciones críticas:
    1. Limpieza de TotalCharges (espacios vacíos a 0)
    2. Casting de tipos (String a Double/Int)
    """
    df_transformed = df.select(
        col("customerid").cast(StringType()),
        col("gender").cast(StringType()),
        col("seniorcitizen").cast(IntegerType()),
        col("partner").cast(StringType()),
        col("dependents").cast(StringType()),
        col("tenure").cast(IntegerType()),
        col("phoneservice").cast(StringType()),
        col("multiplelines").cast(StringType()),
        col("internetservice").cast(StringType()),
        col("onlinesecurity").cast(StringType()),
        col("onlinebackup").cast(StringType()),
        col("deviceprotection").cast(StringType()),
        col("techsupport").cast(StringType()),
        col("streamingtv").cast(StringType()),
        col("streamingmovies").cast(StringType()),
        col("paperlessbilling").cast(StringType()),
        col("paymentmethod").cast(StringType()),
        col("monthlycharges").cast(DoubleType()),
        # LIMPIEZA: Si totalcharges es espacio vacío, poner "0", luego cast a Double
        regexp_replace(col("totalcharges"), "^\\s*$", "0").cast(DoubleType()).alias("totalcharges"),
        col("churn").cast(StringType()),
        col("contract").cast(StringType()) # Columna de partición
    )
    
    if enable_validation:
        # Ejemplo: Filtrar si el ID es nulo o tenure es negativo
        df_transformed = df_transformed.filter(
            (col("customerid").isNotNull()) & (col("tenure") >= 0)
        )
    
    return df_transformed

def insertar_datos_parquet(spark, db_name, table_name, df_transformed):
    table_full_name = f"{db_name}.{table_name}"
    # Usamos saveAsTable para que Spark cree la estructura Parquet automáticamente
    df_transformed.write \
        .mode("overwrite") \
        .format("parquet") \
        .option("compression", "snappy") \
        .partitionBy("contract") \
        .saveAsTable(table_full_name)
    print(f"✅ Datos insertados en {table_full_name}")

# Configuración específica de tu proyecto
TABLAS_CONFIG = [
    {
        "nombre": "customers",
        "partitioned_by": ["contract"],
        "func_calidad": aplicar_reglas_calidad_customers
    }
]

def main():
    args = parse_arguments()
    spark = create_spark_session()
    
    try:
        db_curated = f"{args.env.lower()}_curated"
        db_source = f"{args.env.lower()}_{args.source_db}"
        
        crear_database(spark, args.env.lower(), args.username, args.base_path)
        
        for config in TABLAS_CONFIG:
            table_name = config["nombre"]
            print(f"🔄 Procesando: {table_name}")
            
            # Leer de landing
            df_source = spark.table(f"{db_source}.{table_name}")
            
            # Transformar
            df_transformed = config["func_calidad"](df_source, args.enable_validation)
            
            # Cargar
            insertar_datos_parquet(spark, db_curated, table_name, df_transformed)
            
            # Mostrar resultado
            spark.sql(f"SELECT * FROM {db_curated}.{table_name} LIMIT 5").show()
            
        print("🎉 ¡Capa Curated finalizada con éxito!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
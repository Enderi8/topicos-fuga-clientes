#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script PySpark para despliegue de capa Functional - Caso Fuga de Clientes (Churn)
Adaptado de la arquitectura del docente para el proyecto TOPICOSA
"""

import sys
import argparse
import logging
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import col, when, lit

# =============================================================================
# @section 1. Configuración de logging y parámetros (Sincronizado con Docente)
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Proceso de carga - Capa Functional Project')
    # Estos parámetros son los que usa el spark-submit de tu docente
    parser.add_argument('--env', type=str, default='topicosa', help='Entorno del proyecto')
    parser.add_argument('--username', type=str, default='hadoop', help='Usuario HDFS')
    parser.add_argument('--base_path', type=str, default='/user', help='Ruta base HDFS')
    parser.add_argument('--source_db', type=str, default='curated', help='Base de datos origen')
    parser.add_argument('--num-executors', type=int, default=8)
    parser.add_argument('--executor-memory', type=str, default='2g')
    parser.add_argument('--executor-cores', type=int, default=2)
    parser.add_argument('--enable-broadcast', action='store_true', default=True)
    return parser.parse_args()

# =============================================================================
# @section 2. Inicialización de SparkSession (Tuning del Docente)
# =============================================================================

def create_spark_session(args):
    return SparkSession.builder \
        .appName(f"ProcesoFunctional-Churn-{args.env}") \
        .enableHiveSupport() \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
        .config("hive.exec.dynamic.partition", "true") \
        .config("hive.exec.dynamic.partition.mode", "nonstrict") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.executor.instances", args.num_executors) \
        .config("spark.executor.memory", args.executor_memory) \
        .config("spark.executor.cores", args.executor_cores) \
        .getOrCreate()

# =============================================================================
# @section 3. Funciones de Base de Datos y Tablas
# =============================================================================

def preparar_entorno(spark, db_name, location):
    """Crea la base de datos si no existe"""
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name} LOCATION '{location}'")
    logger.info(f"✅ Database '{db_name}' lista en: {location}")

def crear_tabla_master(spark, db_name, table_name, schema, location):
    """Crea la tabla física en Parquet"""
    columns_def = [f"{f.name} {f.dataType.simpleString().upper()}" for f in schema.fields if f.name != "contract"]
    
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {db_name}.{table_name} (
        {', '.join(columns_def)}
    )
    PARTITIONED BY (contract STRING)
    STORED AS PARQUET
    LOCATION '{location}'
    """
    spark.sql(create_sql)
    logger.info(f"✅ Tabla '{db_name}.{table_name}' registrada")

# =============================================================================
# @section 4. Esquema y Transformación (Feature Engineering Completo)
# =============================================================================

SCHEMA_MASTER = StructType([
    StructField("customerid", StringType(), True),
    StructField("gender_bin", IntegerType(), True),
    StructField("seniorcitizen", IntegerType(), True),
    StructField("partner_bin", IntegerType(), True),
    StructField("dependents_bin", IntegerType(), True),
    StructField("tenure", IntegerType(), True),
    StructField("monthlycharges", DoubleType(), True),
    StructField("totalcharges", DoubleType(), True),
    StructField("label", IntegerType(), True),
    StructField("contract", StringType(), True)
])

def transformar_datos(spark, source_db):
    """Mantiene todas las líneas de binarización que ya tenías"""
    df = spark.table(f"{source_db}.customers")
    
    df_transformed = df.select(
        col("customerid"),
        when(col("gender") == "Female", 1).otherwise(0).alias("gender_bin"),
        col("seniorcitizen").cast(IntegerType()),
        when(col("partner") == "Yes", 1).otherwise(0).alias("partner_bin"),
        when(col("dependents") == "Yes", 1).otherwise(0).alias("dependents_bin"),
        col("tenure").cast(IntegerType()),
        col("monthlycharges").cast(DoubleType()),
        col("totalcharges").cast(DoubleType()),
        when(col("churn") == "Yes", 1).otherwise(0).alias("label"),
        col("contract") 
    )
    return df_transformed

# =============================================================================
# @section 5. Ejecución Principal
# =============================================================================

def main():
    args = parse_arguments()
    spark = create_spark_session(args)
    
    # Construcción dinámica de nombres de BD según los argumentos
    db_curated = f"{args.env.lower()}_{args.source_db}"
    db_functional = f"{args.env.lower()}_functional"
    path_functional = f"{args.base_path}/{args.username}/datalake/{db_functional.upper()}"
    
    try:
        preparar_entorno(spark, db_functional, path_functional)
        
        crear_tabla_master(
            spark, db_functional, "master_churn", 
            SCHEMA_MASTER, f"{path_functional}/master_churn"
        )
        
        df_final = transformar_datos(spark, db_curated)
        df_final.createOrReplaceTempView("v_temp_functional")
        
        insert_sql = f"""
        INSERT OVERWRITE TABLE {db_functional}.master_churn 
        PARTITION(contract) 
        SELECT * FROM v_temp_functional
        """
        spark.sql(insert_sql)
        
        spark.sql(f"MSCK REPAIR TABLE {db_functional}.master_churn")
        logger.info("🚀 PROCESO COMPLETADO EXITOSAMENTE")

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
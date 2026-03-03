#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
import sys
import logging

# Configurar logging profesional
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración de base de datos personalizada
database = 'medalloncustomers' # Según tu preferencia guardada
collection = 'gold_churn'

try:
    # Configuración de Spark para MongoDB
    # IMPORTANTE: Verifica que la IP 172.27.192.1 sea la de tu Host de Windows
    conf = SparkConf() \
        .set("spark.mongodb.connection.uri", "mongodb://172.27.192.1:27017/") \
        .set("spark.mongodb.database", database) \
        .set("spark.mongodb.collection", collection)

    # Crear sesión Spark
    spark = SparkSession.builder \
        .appName("FugaClientes_to_MongoDB") \
        .config(conf=conf) \
        .getOrCreate()

    logger.info("✅ Sesión Spark para MongoDB creada")

    # Leer el CSV final (usamos la ruta de tu proyecto)
    csv_path = "file:/home/hadoop/topicos-fuga-clientes/datalake/gold.csv"
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(csv_path)

    logger.info(f"✅ CSV leído exitosamente: {csv_path}")
    logger.info(f"📊 Total de registros a migrar: {df.count()}")

    # Escribir en MongoDB
    logger.info(f"💾 Migrando datos a MongoDB: {database}.{collection} ...")
    df.write \
        .format("mongodb") \
        .mode("overwrite") \
        .save()

    logger.info("🎉 ¡Migración completada con éxito!")

    # Verificación rápida
    df_verify = spark.read.format("mongodb").load()
    logger.info(f"🔍 Verificación en BD: {df_verify.count()} documentos encontrados.")

except Exception as e:
    logger.error(f"❌ Error en la migración: {str(e)}")
    sys.exit(1)

finally:
    spark.stop()
    logger.info("🛑 Sesión Spark cerrada")
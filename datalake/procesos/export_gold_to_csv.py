#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pyspark.sql import SparkSession

# Crear sesión Spark con soporte Hive
spark = SparkSession.builder \
    .appName("Export-Gold-To-CSV") \
    .enableHiveSupport() \
    .getOrCreate()

# Ajustado a tus bases de datos actuales
# Usamos la tabla que acabas de crear con éxito
database = "topicosa_functional" 
table = "master_churn"

# Leer tabla Hive
df = spark.table(f"{database}.{table}")

# Ruta dentro de TU proyecto en WSL
# He ajustado la ruta para que coincida con tu carpeta 'topicos-fuga-clientes'
output_path = "file:/home/hadoop/topicos-fuga-clientes/datalake/temp"

# Guardar como CSV (coalesce(1) para que sea un solo archivo)
df.coalesce(1) \
  .write \
  .mode("overwrite") \
  .option("header", "true") \
  .option("sep", ",") \
  .csv(output_path)

print(f"✅ Exportación completada en: {output_path}")

spark.stop()
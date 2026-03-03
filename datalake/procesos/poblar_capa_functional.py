#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script PySpark para despliegue de capa Functional (Gold) - Proyecto Telco Churn
Alineado a Objetivos de Negocio: Preguntas Analíticas (Sec 1.4) y KPIs (Sec 1.5)
"""

import sys
import argparse
import traceback
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum as spark_sum, when, round, avg

# =============================================================================
# @section 1. Configuración de parámetros
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description='Proceso de agregación - Capa Functional Churn')
    parser.add_argument('--env', type=str, default='topicosa', help='Entorno: topicosa, topicosb')
    parser.add_argument('--username', type=str, default='hadoop', help='Usuario HDFS')
    parser.add_argument('--base_path', type=str, default='/user', help='Ruta base en HDFS')
    return parser.parse_args()

# =============================================================================
# @section 2. Inicialización de SparkSession
# =============================================================================

def create_spark_session(app_name="Proceso_Functional_Churn_KPIs"):
    return SparkSession.builder \
        .appName(app_name) \
        .enableHiveSupport() \
        .config("hive.exec.dynamic.partition", "true") \
        .config("hive.exec.dynamic.partition.mode", "nonstrict") \
        .getOrCreate()

# =============================================================================
# @section 3. Funciones auxiliares
# =============================================================================

def crear_database(spark, env, username, base_path):
    db_name = f"{env}_functional".lower()
    db_location = f"{base_path}/{username}/datalake/{db_name.upper()}"
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name} LOCATION '{db_location}'")
    return db_name, db_location

def guardar_data_mart(df, db_name, table_name, location):
    print(f"💾 Guardando Data Mart: {table_name}...")
    df.write.format("parquet").mode("overwrite").option("path", f"{location}/{table_name}").saveAsTable(f"{db_name}.{table_name}")

# =============================================================================
# @section 4. Proceso principal
# =============================================================================

def main():
    args = parse_arguments()
    spark = create_spark_session()
    
    try:
        env_lower = args.env.lower()
        db_functional, db_location = crear_database(spark, env_lower, args.username, args.base_path)
        db_source = f"{env_lower}_curated"
        table_source = "customers"
        
        print(f"📥 Leyendo datos limpios desde: {db_source}.{table_source}...")
        df_curated = spark.table(f"{db_source}.{table_source}")
        
        # ---------------------------------------------------------------------
        # PREPARACIÓN BASE (Indicadores base para fórmulas)
        # Calculamos proxy de CLTV (Tenure * MonthlyCharges)
        # ---------------------------------------------------------------------
        df_base = df_curated.withColumn("churn_num", when(col("churn") == "Yes", 1).otherwise(0)) \
                            .withColumn("retencion_num", when(col("churn") == "No", 1).otherwise(0)) \
                            .withColumn("cltv_proxy", col("tenure") * col("monthlycharges"))

        print(f"📊 Generando Tablas para Preguntas Analíticas (Sección 1.4 y 1.5)...")

        # ---------------------------------------------------------------------
        # PREGUNTA 1: Perfil Demográfico (Género, Dependientes, Senior)
        # ---------------------------------------------------------------------
        df_demografico = df_base.groupBy("gender", "dependents", "seniorcitizen").agg(
            count("customerid").alias("total_clientes"),
            spark_sum("churn_num").alias("total_fugas"),
            round((spark_sum("churn_num") / count("customerid")) * 100, 2).alias("tasa_churn_pct"),
            round((spark_sum("retencion_num") / count("customerid")) * 100, 2).alias("tasa_retencion_pct"),
            round(avg("tenure"), 2).alias("tenure_promedio")
        )
        guardar_data_mart(df_demografico, db_functional, "dm_perfil_demografico", db_location)

        # ---------------------------------------------------------------------
        # PREGUNTA 2 y 3: Tipo de contrato y Cargos Mensuales (ARPU y CLTV)
        # ---------------------------------------------------------------------
        df_contratos = df_base.groupBy("contract").agg(
            count("customerid").alias("total_clientes"),
            spark_sum("churn_num").alias("total_fugas"),
            round((spark_sum("churn_num") / count("customerid")) * 100, 2).alias("tasa_churn_pct"),
            round(avg("monthlycharges"), 2).alias("arpu"),
            round(avg("cltv_proxy"), 2).alias("cltv_promedio")
        )
        guardar_data_mart(df_contratos, db_functional, "dm_analisis_contratos", db_location)

        # ---------------------------------------------------------------------
        # PREGUNTA 4: Servicios Adicionales (Seguridad, Soporte Técnico)
        # ---------------------------------------------------------------------
        df_servicios = df_base.groupBy("onlinesecurity", "techsupport").agg(
            count("customerid").alias("total_clientes"),
            spark_sum("churn_num").alias("total_fugas"),
            round((spark_sum("churn_num") / count("customerid")) * 100, 2).alias("tasa_churn_pct")
        )
        # Filtramos los 'No internet service' para ver el impacto real de los servicios
        df_servicios = df_servicios.filter(col("onlinesecurity") != "No internet service")
        guardar_data_mart(df_servicios, db_functional, "dm_impacto_servicios", db_location)

        print("\n🎉 ¡Capa Functional (Gold) generada exitosamente y alineada al negocio!")
        
        # Muestra rápida de validación del KPI de Contratos
        print("\n🔍 Validando respuesta a P2 (Contrato vs Fuga):")
        df_contratos.orderBy(col("tasa_churn_pct").desc()).show()

    except Exception as e:
        print(f"❌ Error durante el proceso: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
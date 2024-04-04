import os

import numpy as np
import pandas as pd

from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T

from pyspark.ml import Transformer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer


class DropNaColumns15Perc(Transformer):
    """
        Выкидывает все колонки у которых больше 15% значений пропущено
    """
    cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage']

    def _transform(self, df: DataFrame) -> DataFrame:
        return df.drop(*self.cols_to_drop)


class FillNaImputer(Transformer):
    """
        Заполняет все пропуски у датасета
    """
    cols_with_none = [
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'MasVnrType', 'Electrical', 'MSZoning', 'Utilities', 'SaleType',
        'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional'
    ]

    cols_with_zero = [
        'GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        'BsmtFullBath', 'BsmtHalfBath'
    ]

    def _transform(self, df: DataFrame) -> DataFrame:
        res_df = df.fillna("None", subset=self.cols_with_none)
        res_df = res_df.fillna(0, subset=self.cols_with_zero)
        return res_df


class IntegerCaster(Transformer):
    """
        Кастит колонки в IntegerType
    """
    cols_to_integer = [
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea'
    ]

    def _transform(self, df: DataFrame) -> DataFrame:
        for col in self.cols_to_integer:
            df = df.withColumn(col, F.col(col).cast(T.IntegerType()))
        return df


class StringIndexerSparkDF(Transformer):
    """
        Индексирует строки превращая их в числа
    """
    def _transform(self, df: DataFrame) -> DataFrame:
        string_columns = []

        for col, dtype in df.dtypes:
            if dtype == 'string':
                string_columns.append(col)

        indexers = [StringIndexer(inputCol=col, outputCol=col+'_index', handleInvalid='keep').fit(df) for col in string_columns]
        pipeline = Pipeline(stages=indexers)
        df_indexed = pipeline.fit(df).transform(df)
        
        return df_indexed


def get_dtype(df: DataFrame, colname: str):
    return [dtype for name, dtype in df.dtypes if name == colname][0]


if __name__ == "__main__":
    session = (
        SparkSession
        .builder
        .appName("PySpark HomeWork_2")
        .master("local")
        .getOrCreate()
    )

    df_pd = pd.read_csv('data/train.csv')
    df = session.createDataFrame(df_pd)

    all_needed_transformers = [
        DropNaColumns15Perc(),
        FillNaImputer(),
        IntegerCaster(),
        StringIndexerSparkDF()
    ]
    for transformer in all_needed_transformers:
        df = transformer.transform(df)

    num_cols_df = []
    for col in df.columns:
        if get_dtype(df, col) != 'string':
            num_cols_df.append(str(col))
    df = df.select(num_cols_df)
    df.show(5)

    vector_assembler = VectorAssembler(inputCols=df.drop("SalePrice").columns, outputCol='features').setHandleInvalid("keep")
    df_vector = vector_assembler.transform(df)
    train, val = df_vector.randomSplit([0.8, 0.2])
    
    lr_model = LinearRegression(featuresCol='features', labelCol='SalePrice', regParam=0.8, elasticNetParam=0.1)
    lr_model = lr_model.fit(train)
    val_predictions = lr_model.transform(val)
    val_predictions.select("features", "SalePrice", "prediction").show(5)

    print(f"Train MSE: {lr_model.summary.meanSquaredError}")
    
    lr_eval = RegressionEvaluator(
        predictionCol="prediction",
        labelCol="SalePrice",
        metricName="mse"
    )
    print(f"Val MSE: {lr_eval.evaluate(val_predictions)}")

import os
from pathlib import Path


SOURCE_QUERY = """
    SELECT 
        age,
        workclass,
        fnlwgt,
        education,
        education_num,
        marital_status,
        occupation,
        relationship,
        race,
        gender,
        capital_gain,
        capital_loss,
        hours_per_week,
        native_country,
        CASE WHEN income_bracket = ' <=50K' THEN 0 ELSE 1 END AS income_bracket
    FROM 
        @dataset_name.census
    WHERE
        age <= @age
"""


def generate_source_query(dataset_name:str ='sample_datasets', age:int=1000)->str:
    sql = SOURCE_QUERY.replace(
        '@age', str(age)).replace(
        '@dataset_name',dataset_name)
    return sql

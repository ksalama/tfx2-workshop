# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SQL Utilities"""

import os
from pathlib import Path


SOURCE_QUERY = """
    SELECT 
        age,
        TRIM(workclass) AS workclass,
        fnlwgt,
        TRIM(education) AS education,
        education_num,
        TRIM(marital_status) AS marital_status,
        TRIM(occupation) AS occupation,
        TRIM(relationship) AS relationship,
        TRIM(race) AS race,
        TRIM(gender) AS gender,
        capital_gain,
        capital_loss,
        hours_per_week,
        TRIM(native_country) AS native_country,
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

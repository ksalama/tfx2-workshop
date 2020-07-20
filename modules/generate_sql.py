import os
from pathlib import Path


def generate_source_query(age:int=1000)->str:
    sql_script_path = 'modules/source_query.sql'
    with open(sql_script_path, "r") as sql_file:
        sql = sql_file.read()
    sql = sql.replace('@age', str(age))
    return sql

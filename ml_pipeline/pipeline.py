# Copyright 2020 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Census training pipeline DSL."""

import os
from typing import Dict, List, Text, Optional
from kfp import gcp
from tfx.orchestration import data_types
from tfx.components.base import executor_spec
from tfx.components.example_gen.big_query_example_gen.component import BigQueryExampleGen
from tfx.components.common_nodes.importer_node import ImporterNode
from tfx.components.example_validator.component import ExampleValidator
from tfx.components.schema_gen.component import SchemaGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.transform.component import Transform
from tfx.components.trainer.component import Trainer
from tfx.components.evaluator.component import Evaluator
from modules.custom_components import AccuracyModelValidator
from tfx.components.pusher.component import Pusher
from tfx.components.trainer import executor as trainer_executor
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.extensions.google_cloud_ai_platform.pusher import executor as ai_platform_pusher_executor
from tfx.orchestration import pipeline
from tfx.types.standard_artifacts import Schema

from modules import sql_utils, helper

RAW_SCHEMA_DIR='raw_schema'
TRANSFORM_MODULE_FILE='modules/transform.py'
TRAIN_MODULE_FILE='modules/train.py'


def create_pipeline(pipeline_name: Text, 
                    pipeline_root: Text, 
                    dataset_name: data_types.RuntimeParameter,
                    train_steps: data_types.RuntimeParameter,
                    eval_steps: data_types.RuntimeParameter,
                    accuracy_threshold: data_types.RuntimeParameter,
                    ai_platform_training_args: Dict[Text, Text],
                    ai_platform_serving_args: Dict[Text, Text],
                    beam_pipeline_args: List[Text],
                    enable_cache: Optional[bool] = False) -> pipeline.Pipeline:
    """Implements the online news pipeline with TFX."""

    # Dataset, table and/or 'where conditions' can be passed as pipeline args.
    query=sql_utils.generate_source_query(dataset_name=str(dataset_name)) 
    
    # Brings data into the pipeline from BigQuery.
    example_gen = BigQueryExampleGen(
        query=query
    )

    # Computes statistics over data for visualization and example validation.
    statistics_gen = StatisticsGen(input_data=example_gen.outputs.examples)

    # Import schema from local directory.
    schema_importer = ImporterNode(
        instance_name='RawSchemaImporter',
        source_uri=RAW_SCHEMA_DIR,
        artifact_type=Schema,
    )

    # Performs anomaly detection based on statistics and data schema.
    validate_stats = ExampleValidator(
        stats=statistics_gen.outputs.output, 
        schema=schema_importer.outputs.result
    )

    # Performs transformations and feature engineering in training and serving.
    transform = Transform(
        input_data=example_gen.outputs.examples,
        schema=schema_importer.outputs.result,
        module_file=TRANSFORM_MODULE_FILE
    )

    # Train and export serving and evaluation saved models.
#     trainer = Trainer(
#         custom_executor_spec=executor_spec.ExecutorClassSpec(
#             ai_platform_trainer_executor.GenericExecutor),
#         module_file=TRAIN_MODULE_FILE,
#         transformed_examples=transform.outputs.transformed_examples,
#         schema=schema_importer.outputs.result,
#         transform_output=transform.outputs.transform_output,
#         train_args={'num_steps': train_steps},
#         eval_args={'num_steps': eval_steps},
#         custom_config={'ai_platform_training_args': ai_platform_training_args}
#     )
    
    trainer = Trainer(
        custom_executor_spec=executor_spec.ExecutorClassSpec(
            trainer_executor.GenericExecutor),
        module_file=TRAIN_MODULE_FILE,
        transformed_examples=transform.outputs.transformed_examples,
        schema=schema_importer.outputs.result,
        transform_graph=transform.outputs.transform_graph,
        train_args={'num_steps': train_steps},
        eval_args={'num_steps': eval_steps},
    )

    # Uses TFMA to compute a evaluation statistics over features of a model.
    model_analyzer = Evaluator(
        examples=example_gen.outputs.examples,
        model=trainer.outputs.model,
        eval_config=helper.get_eval_config()
    )

    # Use a custom AccuracyModelValidator component to validate the model.
    model_validator = AccuracyModelValidator(
        eval_results=model_analyzer.outputs.output,
        model=trainer.outputs.model,
        accuracy_threshold=accuracy_threshold,
        slice_accuracy_tolerance=0.15,
    )

    # Checks whether the model passed the validation steps and pushes the model
    # to its destination if check passed.
    pusher = Pusher(
        custom_executor_spec=executor_spec.ExecutorClassSpec(
            ai_platform_pusher_executor.Executor),
        model_export=trainer.outputs.output,
        model_blessing=model_validator.outputs.blessing,
        custom_config={'ai_platform_serving_args': ai_platform_serving_args}
    )
    
    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            example_gen, statistics_gen, schema_importer, validate_stats, transform,
            trainer, model_analyzer, model_validator, pusher
      ],
      enable_cache=enable_cache,
      beam_pipeline_args=beam_pipeline_args)

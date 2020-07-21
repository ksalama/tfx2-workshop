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
"""KFP runner"""

import kfp

from kfp import gcp
from tfx.orchestration import data_types
from tfx.orchestration.kubeflow import kubeflow_dag_runner

from typing import Optional, Dict, List, Text

import config
import pipeline

if __name__ == '__main__':

    # Set the values for the compile time parameters
    
    ai_platform_training_args = {
        'project': config.PROJECT_ID,
        'region': config.GCP_REGION,
        'masterConfig': {
            'imageUri': config.TFX_IMAGE
        }
      }

    ai_platform_serving_args = {
        'project_id': config.PROJECT_ID,
        'model_name': config.MODEL_NAME,
        'runtimeVersion': config.RUNTIME_VERSION,
        'pythonVersion': config.PYTHON_VERSION,
        'regions': [config.GCP_REGION]
    }

    beam_tmp_folder = '{}/beam/tmp'.format(config.ARTIFACT_STORE_URI)
    
    beam_pipeline_args = [
        '--runner=' + config.BEAM_RUNNER,
        '--experiments=shuffle_mode=auto',
        '--project=' + config.PROJECT_ID,
        '--temp_location=' + beam_tmp_folder,
        '--region=' + config.GCP_REGION,
    ]
    
  
    # Set the default values for the pipeline runtime parameters

    train_steps = data_types.RuntimeParameter(
        name='train-steps',
        default=5000,
        ptype=int
    )
    
    eval_steps = data_types.RuntimeParameter(
        name='eval-steps',
        default=500,
        ptype=int
    )
    
    accuracy_threshold = data_types.RuntimeParameter(
        name='accuracy-threshold',
        default=0.75,
        ptype=float
    )
    
    pipeline_root = '{}/{}/{}'.format(
        config.ARTIFACT_STORE_URI, 
        config.PIPELINE_NAME,
        kfp.dsl.RUN_ID_PLACEHOLDER
    )

    # Set KubeflowDagRunner settings
    metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()

    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        kubeflow_metadata_config = metadata_config,
        pipeline_operator_funcs = kubeflow_dag_runner.get_default_pipeline_operator_funcs(
            config.USE_KFP_SA == 'True'),
         tfx_image=config.TFX_IMAGE)

    # Compile the pipeline
    kubeflow_dag_runner.KubeflowDagRunner(config=runner_config).run(
        pipeline.create_pipeline(
            pipeline_name=config.PIPELINE_NAME,
            pipeline_root=pipeline_root,
            dataset_name=config.DATASET_NAME,
            train_steps=train_steps,
            eval_steps=eval_steps,
            accuracy_threshold=accuracy_threshold,
            ai_platform_training_args=ai_platform_training_args,
            ai_platform_serving_args=ai_platform_serving_args,
            beam_pipeline_args=beam_pipeline_args))
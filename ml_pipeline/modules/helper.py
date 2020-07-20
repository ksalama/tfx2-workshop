# Copyright 2020 Google Inc. All Rights Reserved.
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
"""Helper functions."""

import tensorflow_model_analysis as tfma

def get_eval_config():
    model_specs = [
        tfma.ModelSpec(
            signature_name='serving_tf_examples',
            label_key='income_bracket',
            example_weight_key='fnlwgt')]

    metrics_specs = [
        tfma.MetricsSpec(
            metrics = [
                tfma.MetricConfig(class_name='BinaryAccuracy'),
                tfma.MetricConfig(class_name='ExampleCount')])]

    slicing_specs = [
            tfma.SlicingSpec(),
            tfma.SlicingSpec(feature_keys=['occupation'])]

    eval_config = tfma.EvalConfig(
        model_specs=model_specs,
        metrics_specs=metrics_specs,
        slicing_specs=slicing_specs
    )
    
    return eval_config
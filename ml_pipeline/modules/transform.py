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
"""TFT Preprocessing."""

import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_data_validation as tfdv

TARGET_FEATURE_NAME = 'income_bracket'
WEIGHT_FEATURE_NAME = 'fnlwgt'
RAW_SCHEMA_LOCATION = 'ml_pipeline/raw_schema/schema.pbtxt'

raw_schema = tfdv.load_schema_text(RAW_SCHEMA_LOCATION)

def _prep(feature):
    #return tf.squeeze(feature, axis=1)
    return feature

def preprocessing_fn(input_features):

    processed_features = {}

    for feature in raw_schema.feature:
        
        # Pass the target feature as is.
        if feature.name in [TARGET_FEATURE_NAME, WEIGHT_FEATURE_NAME]:
            processed_features[feature.name] = _prep(input_features[feature.name])
            continue
            
        if feature.type == 1:
            # Extract vocabulary and integerize categorical features.
            processed_features[feature.name+"_integerized"] = _prep(tft.compute_and_apply_vocabulary(
                input_features[feature.name], vocab_filename=feature.name))
        else:
            # normalize numeric features.
            processed_features[feature.name+"_scaled"] = _prep(tft.scale_to_z_score(input_features[feature.name]))

        # Bucketize age using quantiles. 
        quantiles = tft.quantiles(input_features["age"], num_buckets=5, epsilon=0.01)
        processed_features["age_bucketized"] = _prep(tft.apply_buckets(
            input_features["age"], bucket_boundaries=quantiles))

    return processed_features

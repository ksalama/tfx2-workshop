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
"""Training Keras Model."""

import math
import os
import tensorflow as tf
import tensorflow_transform as tft

TARGET_FEATURE_NAME = 'income_bracket'
WEIGHT_FEATURE_NAME = 'fnlwgt'
TARGET_FEATURE_LABELS = ['<=50K', '>50K']

LABEL_KEY = 'predicted_label'
SCORE_KEY = 'confidence'
PROBABILITIES_KEY = 'probabilities'
SERVING_SIGNATURE_NAME = 'serving_features'
EVAL_SIGNATURE_NAME = 'serving_default'

HIDDEN_UNITS = [32, 32]
LEARNING_RATE = 0.0001
BATCH_SIZE=128
DROPOUT_RATE=0.15
TRAIN_SIZE=48840


# Input function for train and eval transformed data
def create_dataset(file_pattern, transform_output,
                  batch_size=128, shuffle=False):
    
    def _gzip_reader_fn(filenames):
        return tf.data.TFRecordDataset(
            filenames, compression_type='GZIP')
    
    transformed_feature_spec = (
        transform_output.transformed_feature_spec().copy())
    
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=TARGET_FEATURE_NAME,
        shuffle=shuffle,
        shuffle_buffer_size=(5 * batch_size),
        num_epochs=1
    )
    
    return dataset.cache()



# Create feature columns
def create_feature_columns(transform_output):
    
    feature_columns = []
    
    # Create feature columns based on the transform schema
    transformed_features = transform_output.transformed_metadata.schema.feature
    for feature in transformed_features:
        
        if feature.name in [TARGET_FEATURE_NAME, WEIGHT_FEATURE_NAME]:
            continue

        if hasattr(feature, 'int_domain') and feature.int_domain.is_categorical:
            vocab_size = feature.int_domain.max + 1
            feature_columns.append(
                tf.feature_column.embedding_column(
                    tf.feature_column.categorical_column_with_identity(
                        feature.name, num_buckets=vocab_size),
                    dimension = int(math.sqrt(vocab_size))))
        else:
            feature_columns.append(
                tf.feature_column.numeric_column(feature.name))

    return feature_columns



# Create keras model
def create_keras_model(params, feature_columns):

    layers = []
    layers.append(tf.keras.layers.DenseFeatures(feature_columns))
    for units in HIDDEN_UNITS:
        layers.append(tf.keras.layers.Dense(units=units, activation='relu'))
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.Dropout(rate=DROPOUT_RATE))
    
    layers.append(tf.keras.layers.Dense(units=1, activation='sigmoid'))
  
    model = tf.keras.Sequential(layers=layers, name='census_classifier')
    model.output_names=['output']
    
    adam_optimzer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=adam_optimzer, 
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), 
        metrics=[
            tf.keras.metrics.BinaryAccuracy(), 
            tf.keras.metrics.AUC(curve="ROC"),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.SensitivityAtSpecificity(0.5),
            #tfa.metrics.F1Score(num_classes=1)
        ], 
        loss_weights=None)
    
    return model



def _make_tf_examples_eval_fn(model, transform_output):
    
    model.tft_layer = transform_output.transform_features_layer()
    
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        
        feature_spec = transform_output.raw_feature_spec()
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        
        transformed_features = model.tft_layer(parsed_features)
        transformed_features.pop(TARGET_FEATURE_NAME)
        transformed_features.pop(WEIGHT_FEATURE_NAME)

        probabilities =  model(transformed_features)
        return probabilities

    return serve_tf_examples_fn


def _make_features_serving_fn(model, transform_output):
    
    model.tft_layer = transform_output.transform_features_layer()

    @tf.function
    def serve_features_fn(features):
        
        transformed_features = model.tft_layer(features)
        
        probabilities = model(transformed_features)
        labels = tf.constant(TARGET_FEATURE_LABELS, dtype=tf.string)
        predicted_class_indices = tf.argmax(probabilities, axis=1)
        predicted_class_label = tf.gather(
            params=labels, indices=predicted_class_indices)
        prediction_confidence = tf.reduce_max(probabilities, axis=1)
    
        return {
            LABEL_KEY: predicted_class_label,
            SCORE_KEY:prediction_confidence,
            PROBABILITIES_KEY: probabilities}

    return serve_features_fn



def create_model_signatures(model, transform_output):
    
    raw_feature_spec = transform_output.raw_feature_spec()
    raw_feature_spec.pop(TARGET_FEATURE_NAME)
    raw_feature_spec.pop(WEIGHT_FEATURE_NAME)
    
    features_input_signature = {
        feature: tf.TensorSpec(shape=(None, 1), dtype=spec.dtype, name=feature)
        for feature, spec in raw_feature_spec.items()}
    
    serving_fn = _make_features_serving_fn(model, transform_output)
    serving_signature = serving_fn.get_concrete_function(features_input_signature) 
    
    
    eval_fn = _make_tf_examples_eval_fn(model, transform_output)
    eval_signature = eval_fn.get_concrete_function(
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples'))

    signatures = {        
        SERVING_SIGNATURE_NAME: serving_signature,
        EVAL_SIGNATURE_NAME: eval_signature
    }
    
    return signatures
    

# TFX will call this function
def run_fn(params):

    transform_output = tft.TFTransformOutput(params.transform_output)
    
    train_dataset = create_dataset(
        params.train_files, transform_output, BATCH_SIZE, shuffle=True)
    
    eval_dataset = create_dataset(
        params.eval_files, transform_output, BATCH_SIZE)
    
    feature_columns = create_feature_columns(transform_output)
    
    model = create_keras_model(params, feature_columns)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(os.path.dirname(params.serving_model_dir), 'logs'), 
        update_freq='batch')
    
    earlystopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_binary_accuracy', 
        patience=5,
        restore_best_weights=True
    )
    
    callbacks = [tensorboard_callback, earlystopping_callback]

    steps_per_epoch = int(math.ceil(TRAIN_SIZE / BATCH_SIZE))
    epochs = int(params.train_steps/steps_per_epoch)
    
    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=eval_dataset,
        validation_steps=params.eval_steps,
        callbacks=callbacks
    )
    
    signatures = create_model_signatures(model, transform_output)
  
    model.save(
        params.serving_model_dir, 
        save_format='tf', 
        signatures=signatures
    )

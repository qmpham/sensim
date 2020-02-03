# coding=utf-8
# Copyright 2019-present, Facebook, Inc and the HuggingFace Inc. team.
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
""" TF 2.0 XLM model.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
import os
import sys
import itertools
import numpy as np
import tensorflow as tf
from transformers.configuration_xlm import *
from transformers.modeling_tf_utils import *
from transformers.modeling_tf_xlm import *
from transformers.file_utils import add_start_docstrings

class TFXLMForSequenceEmbedding(TFXLMPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.transformer = TFXLMMainLayer(config, name='transformer')
        
        self.src_forward_layer = tf.keras.layers.LSTM(512, activation='relu', return_sequences=True, go_backwards=False, return_state=True)
        self.src_backward_layer = tf.keras.layers.LSTM(512, activation='relu', return_sequences=True, go_backwards=True, return_state=True)
        self.src_encoder = tf.keras.layers.Bidirectional(self.src_forward_layer, backward_layer=self.src_backward_layer)

        self.tgt_forward_layer = tf.keras.layers.LSTM(512, activation='relu', return_sequences=True, go_backwards=False, return_state=True)
        self.tgt_backward_layer = tf.keras.layers.LSTM(512, activation='relu', return_sequences=True, go_backwards=True, return_state=True)
        self.tgt_encoder = tf.keras.layers.Bidirectional(self.tgt_forward_layer, backward_layer=self.tgt_backward_layer)
        self.config = {"aggr":"sum"}
    @property
    def dummy_inputs(self):
        return ({"input_ids":tf.constant(DUMMY_INPUTS), "langs": tf.constant([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]), "lengths": tf.constant([5,5,5])},
                {"input_ids":tf.constant(DUMMY_INPUTS), "langs": tf.constant([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]), "lengths": tf.constant([5,5,5])})
    
    def call(self, inputs, sign_src=1.0, sign_tgt=1.0, src_padding_mask=None, tgt_padding_mask=None, training=None, **kwargs):       
        src_inputs = inputs[0]
        tgt_inputs = inputs[1]
        src_transformer_outputs = self.transformer(src_inputs, **kwargs)
        tgt_transformer_outputs = self.transformer(tgt_inputs, **kwargs)
        #src_output = src_transformer_outputs[0]
        #tgt_output = tgt_transformer_outputs[0]
        #print(self.src_encoder(src_output, mask=src_padding_mask, training=training))
        #src, _, _, _, _ = self.src_encoder(src_output, mask=src_padding_mask, training=training)
        #tgt, _, _, _, _ = self.tgt_encoder(tgt_output, mask=tgt_padding_mask, training=training) 
        src = src_transformer_outputs
        tgt = tgt_transformer_outputs
        src =tf.nn.dropout(src, 0.1)
        tgt =tf.nn.dropout(tgt, 0.1)
        self.align = tf.map_fn(lambda x: tf.matmul(x[0], tf.transpose(x[1])), (src, tgt), dtype=tf.float32, name="align")  
            
        R = 1.0
        if self.config["aggr"] == "lse":
            self.aggregation_src = tf.divide(tf.math.log(tf.map_fn(lambda xl: tf.reduce_sum(xl[0][:xl[1], :], 0),
                                                (tf.exp(tf.transpose(self.align, [0, 2, 1]) * R), tgt_inputs["lengths"]),
                                                dtype=tf.float32)), R, name="aggregation_src")
            self.aggregation_tgt = tf.divide(tf.math.log(tf.map_fn(lambda xl: tf.reduce_sum(xl[0][:xl[1], :], 0),
                                                    (tf.exp(self.align * R), src_inputs["lengths"]), dtype=tf.float32)),
                                                R, name="aggregation_tgt")
        elif self.config["aggr"] == "sum":
            self.aggregation_src = tf.map_fn(lambda xl: tf.reduce_sum(xl[0][:xl[1], :], 0),
                                                (tf.transpose(self.align, [0, 2, 1]), tgt_inputs["lengths"]),
                                                dtype=tf.float32, name="aggregation_src")
            self.aggregation_tgt = tf.map_fn(lambda xl: tf.reduce_sum(xl[0][:xl[1], :], 0),
                                                (self.align, src_inputs["lengths"]),
                                                dtype=tf.float32, name="aggregation_tgt")
        elif self.config["aggr"] == "max":
            self.aggregation_src = tf.map_fn(lambda xl: tf.reduce_max(xl[0][:xl[1], :], 0),
                                                (tf.transpose(self.align, [0, 2, 1]), tgt_inputs["lengths"]),
                                                dtype=tf.float32, name="aggregation_src")
            self.aggregation_tgt = tf.map_fn(lambda xl: tf.reduce_max(xl[0][:xl[1], :], 0),
                                                (self.align, src_inputs["lengths"]),
                                                dtype=tf.float32, name="aggregation_tgt")
        else:
            sys.stderr.write("error: bad aggregation option '{}'\n".format(self.config.aggr))
            sys.exit(1)
         
        self.output_src = tf.math.log(1 + tf.exp(self.aggregation_src * sign_src))
        self.output_tgt = tf.math.log(1 + tf.exp(self.aggregation_tgt * sign_tgt))
        self.loss_src = tf.reduce_mean(tf.map_fn(lambda xl: tf.reduce_sum(xl[0][:xl[1]]),
                                                         (self.output_src, src_inputs["lengths"]), dtype=tf.float32))
        self.loss_tgt = tf.reduce_mean(tf.map_fn(lambda xl: tf.reduce_sum(xl[0][:xl[1]]),
                                                         (self.output_tgt, tgt_inputs["lengths"]), dtype=tf.float32))
        self.loss = self.loss_tgt + self.loss_src

        return self.align, self.aggregation_src, self.aggregation_tgt, self.loss

    def encode(self, inputs, padding_mask, lang="en"):
      """
      if lang == "en":
        transformer_outputs = self.transformer(inputs)
        output = transformer_outputs[0]
        _, _, _, fw_last, bw_last = self.src_encoder(output, mask=padding_mask, training=False)
        return tf.concat([fw_last, bw_last],1)
      elif lang == "fr":
        transformer_outputs = self.transformer(inputs)
        output = transformer_outputs[0]
        _ , _, _, fw_last, bw_last = self.tgt_encoder(output, mask=padding_mask, training=False)
        return tf.concat([fw_last, bw_last],1)
      else:
        sys.stderr.write("error: bad language option '{}'\n".format("en, fr"))
        sys.exit(1)
      """
      return tf.reduce_mean(self.transformer(inputs),1)
    
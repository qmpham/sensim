import tensorflow as tf
import torch
from torch import nn
from transformers import *
from transformers import pipeline, glue_convert_examples_to_features
from data_loader import training_pipeline, process_fn_, Dataset, function_on_next
from model import TFXLMForSequenceEmbedding, TFXLMForSequenceClassification
import argparse
import logging
import yaml
import os
import tensorflow_addons as tfa
from optimization import *
import numpy as np
import faiss
import sklearn 
tf.get_logger().setLevel(logging.INFO)

def build_mask(self, inputs, sequence_length=None, dtype=tf.bool):
    """Builds a boolean mask for :obj:`inputs`."""
    if sequence_length is None:
      return None
    return tf.sequence_mask(sequence_length, maxlen=tf.shape(inputs)[1], dtype=dtype)

def evaluate(model, config, checkpoint_manager, checkpoint, ckpt_path, model_name_or_path, tokenizer_class, tokenizer_cache_dir):
  if ckpt_path == None:
    ckpt_path = checkpoint_manager.latest_checkpoint
  tf.get_logger().info("Evaluating model %s", ckpt_path)  
  checkpoint.restore(ckpt_path)
  validation_dataset = Dataset(config.get("validation_file_path",None),   
              os.path.join(config.get("model_dir"),"data"),
              config.get("seq_size"), 
              config.get("max_sents"), 
              config.get("do_shuffle"), 
              config.get("do_skip_empty"),
              procedure="dev",
              model_name_or_path = model_name_or_path,
              tokenizer_class = tokenizer_class,
              tokenizer_cache_dir = tokenizer_cache_dir)
  iterator = iter(validation_dataset.create_one_epoch(do_shuffle=False, mode="p")) 

  @tf.function
  def encode_next():    
    src, tgt = next(iterator)
    padding_mask = build_mask(src["input_ids"], src["lengths"])
    src_sentence_embedding = model.encode(src, padding_mask)
    padding_mask = build_mask(tgt["input_ids"], tgt["lengths"])
    tgt_sentence_embedding = model.encode(tgt, padding_mask)
    return src_sentence_embedding, tgt_sentence_embedding
  # Iterates on the dataset.  
  src_sentence_embedding_list = []
  tgt_sentence_embedding_list = []
  while True:    
    try:
      src_sentence_embedding_, tgt_sentence_embedding_ = encode_next()
      src_sentence_embedding_list.append(src_sentence_embedding_.numpy())
      tgt_sentence_embedding_list.append(tgt_sentence_embedding_.numpy())      
    except tf.errors.OutOfRangeError:
      break
  src_sentences = np.concatenate(src_sentence_embedding_list, axis=0)
  tgt_sentences = np.concatenate(tgt_sentence_embedding_list, axis=0)
  print("src_sentences",src_sentences.shape)
  print("tgt_sentences",tgt_sentences.shape)
  d = src_sentences.shape[-1]
  index = faiss.IndexFlatIP(d)   # build the index
  print("faiss state: ", index.is_trained)
  index.add(src_sentences)       # add vectors to the index
  print("number of sentences: %d"%index.ntotal)
  k = 1
  D, I = index.search(tgt_sentences, k)     # tgt -> src search  
  print(sklearn.metrics.accuracy_score(np.arange(index.ntotal), I))
  
def encode():
  return

def train(strategy, config):
  #####
  config_class, model_class, tokenizer_class = (XLMConfig, TFXLMForSequenceEmbedding, XLMTokenizer)
  model_name_or_path = 'xlm-mlm-enfr-1024'
  config_cache_dir = os.path.join(config.get("model_dir"),"config")
  model_cache_dir = os.path.join(config.get("model_dir"),"model")
  tokenizer_cache_dir = os.path.join(config.get("model_dir"),"tokenizer")
  #####
  train_dataset = Dataset(config.get("filepath",None),  
              config.get("training_data_save_path"),
              config.get("seq_size"), 
              config.get("max_sents"), 
              config.get("do_shuffle"), 
              config.get("do_skip_empty"),
              model_name_or_path = model_name_or_path,
              tokenizer_class = tokenizer_class,
              tokenizer_cache_dir = tokenizer_cache_dir)
  pretrained_config = config_class.from_pretrained(
      model_name_or_path,    
      cache_dir=config_cache_dir if config_cache_dir else None)
  with strategy.scope():  
    model = model_class.from_pretrained(
      model_name_or_path,
      config=pretrained_config,
      cache_dir=model_cache_dir if model_cache_dir else None)  
    ##### Optimizers
    learning_rate = ScheduleWrapper(schedule=NoamDecay(scale=1.0, model_dim=512, warmup_steps=config.get("warmup_steps", 4000)), step_duration= config.get("step_duration",16))
    optimizer = tfa.optimizers.LazyAdam(learning_rate)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)     
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, config["model_dir"], max_to_keep=5)
    if checkpoint_manager.latest_checkpoint is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
      checkpoint.restore(checkpoint_manager.latest_checkpoint)
      checkpoint_path = checkpoint_manager.latest_checkpoint
  #####
  ##### Training functions
  with strategy.scope():    
    gradient_accumulator = GradientAccumulator()  

  def _accumulate_gradients(src, tgt, sign):
    src_padding_mask = build_mask(src["input_ids"],src["lengths"])
    tgt_padding_mask = build_mask(tgt["input_ids"],tgt["lengths"])
    align, aggregation_src, aggregation_tgt, loss = model((src,tgt),sign_src=sign, sign_tgt=sign, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask, training=True)
    #tf.print("aggregation_src", aggregation_src, "aggregation_tgt", aggregation_tgt, "sign", sign, summarize=1000)
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    gradients = optimizer.get_gradients(loss, variables)
    #gradients = [(tf.clip_by_norm(grad, 0.1)) for grad in gradients]
    gradient_accumulator(gradients)
    num_examples = tf.shape(src["input_ids"])[0]
    return loss, num_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      scaled_gradient = gradient / 2.0
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
  
  u_epoch_dataset = train_dataset.create_one_epoch(mode="u")
  p_epoch_dataset = train_dataset.create_one_epoch(mode="p")

  @function_on_next(u_epoch_dataset)
  def _u_train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target, 1.0))
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  @function_on_next(p_epoch_dataset)
  def _p_train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target, -1.0))
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  #### Training  

  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  report_every = config.get("report_every", 100)
  save_every = config.get("save_every", 1000)
  eval_every = config.get("eval_every", 1000)
  train_steps = config.get("train_steps", 100000)

  u_training_flow = iter(_u_train_forward())
  p_training_flow = iter(_p_train_forward())

  p_losses = []
  u_losses = []
  _number_examples = []
  import time
  start = time.time()
  with _summary_writer.as_default():
    while True:    
        try:
          u_loss, u_examples_num = next(u_training_flow)
          p_loss, p_examples_num = next(p_training_flow)
          _step()          
          p_losses.append(p_loss)
          u_losses.append(u_loss)
          
          _number_examples.extend([u_examples_num, p_examples_num])
          step = optimizer.iterations.numpy()
          if step % report_every == 0:
            elapsed = time.time() - start
            tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; u_loss = %f; p_loss = %f, number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(u_losses), np.mean(p_losses), np.sum(_number_examples), elapsed)
            start = time.time()
            u_losses = []
            p_losses = []
            _number_examples = []
          if step % save_every == 0:
            tf.get_logger().info("Saving checkpoint for step %d", step)
            checkpoint_manager.save(checkpoint_number=step)
          if step % eval_every == 0:
            ckpt_path = None
            evaluate(model, config, checkpoint_manager, checkpoint, ckpt_path, model_name_or_path, tokenizer_class, tokenizer_cache_dir)
          tf.summary.flush()
          if step > train_steps:
            break
        except StopIteration: #tf.errors.OutOfRangeError:
          print("next epoch")
          u_epoch_dataset = train_dataset.create_one_epoch(mode="u")
          p_epoch_dataset = train_dataset.create_one_epoch(mode="p")
          @function_on_next(u_epoch_dataset)
          def _u_train_forward(next_fn):    
            with strategy.scope():
              per_replica_source, per_replica_target = next_fn()
              per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
                  _accumulate_gradients, args=(per_replica_source, per_replica_target, 1.0))
              loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
              num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
            return loss, num_examples

          @function_on_next(p_epoch_dataset)
          def _p_train_forward(next_fn):    
            with strategy.scope():
              per_replica_source, per_replica_target = next_fn()
              per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
                  _accumulate_gradients, args=(per_replica_source, per_replica_target, -1.0))
              loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
              num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
            return loss, num_examples
          
          u_training_flow = iter(_u_train_forward())
          p_training_flow = iter(_p_train_forward())

def main():

  devices = tf.config.experimental.list_logical_devices(device_type="GPU")
  print(devices)
  strategy = tf.distribute.MirroredStrategy(devices=[d.name for d in devices])
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("run", choices=["train", "debug", "encode" ], help="Run type.")
  parser.add_argument("--config", required=True , help="configuration file")
  parser.add_argument("--file")
  parser.add_argument("--ckpt", default=None)
  parser.add_argument("--output", default="sentembedding")

  args = parser.parse_args()
  print("Running mode: ", args.run)
  config_file = args.config
  with open(config_file, "r") as stream:
      config = yaml.load(stream)  

  if args.run == "train":
    train(strategy, config)
  elif args.run == "encode":
    encode() 

if __name__ == "__main__":
  main()

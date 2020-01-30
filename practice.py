import tensorflow as tf
import torch
from torch import nn
from transformers import *
from transformers import pipeline, glue_convert_examples_to_features
import tensorflow_datasets
from data_loader import training_pipeline, process_fn_
from model import TFXLMForSequenceEmbedding
data = tf.data.TextLineDataset('/home/minhquang/reference/UFAL.med.en-fr.tst.en')
#### xlm model configuration
# vocab_size=30145,
# emb_dim=2048,
# n_layers=12,
# n_heads=16,
# dropout=0.1,
# attention_dropout=0.1,
# gelu_activation=True,
# sinusoidal_embeddings=False,
# causal=False,
# asm=False,
# n_langs=1,
# use_lang_emb=True,
# max_position_embeddings=512,
# embed_init_std=2048 ** -0.5,
# layer_norm_eps=1e-12,
# init_std=0.02,
# bos_index=0,
# eos_index=1,
# pad_index=2,
# unk_index=3,
# mask_index=5,
# is_encoder=True,
# summary_type='first',
# summary_use_proj=True,
# summary_activation=None,
# summary_proj_to_labels=True,
# summary_first_dropout=0.1,
# start_n_top=5,
# end_n_top=5,
###

config_class, model_class, tokenizer_class = (XLMConfig, TFXLMForSequenceEmbedding, XLMTokenizer)
model_name_or_path = 'xlm-mlm-enfr-1024'
cache_dir = "models/xlm"
config_name = None
tokenizer_name = None
config = config_class.from_pretrained(
    model_name_or_path,    
    cache_dir=cache_dir if cache_dir else None)
tokenizer = tokenizer_class.from_pretrained(
    model_name_or_path,
    cache_dir=cache_dir if cache_dir else None)
model = model_class.from_pretrained(
    model_name_or_path,
    config=config,
    cache_dir=cache_dir if cache_dir else None)
print(tokenizer.encode("he is going", add_special_tokens=True))
####
process_fn = process_fn_(tokenizer)
dataset = tf.data.Dataset.zip((tf.data.TextLineDataset('test.en'),tf.data.TextLineDataset('test.fr')))
batch_size = 10  
train_dataset = dataset.apply(training_pipeline(batch_size,
                  batch_type="examples",
                  batch_multiplier=1,
                  batch_size_multiple=1,
                  process_fn=process_fn,
                  length_bucket_width=None,
                  features_length_fn = lambda src: tf.shape(src["input_ids"])[0],
                  labels_length_fn = lambda tgt: tf.shape(tgt["input_ids"])[0],
                  maximum_features_length=50,
                  maximum_labels_length=50,
                  single_pass=False,
                  num_shards=1,
                  shard_index=0,
                  num_threads=None,
                  shuffle_buffer_size=None,
                  prefetch_buffer_size=200))
for element in train_dataset:
  print(element)
  print(tokenizer.decode(element[0]["input_ids"][0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
  print(tokenizer.decode(element[1]["input_ids"][0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
  print(model(element))

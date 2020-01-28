import tensorflow as tf
import torch
from torch import nn
from transformers import *
from transformers import pipeline, glue_convert_examples_to_features
import tensorflow_datasets
from data_loader import training_pipeline, process_fn_
# Transformers has a unified API
# for 10 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut
MODELS = [(TFXLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024')]

cos = nn.CosineSimilarity(dim=0, eps=1e-6)
data = tf.data.TextLineDataset('/home/minhquang/reference/UFAL.med.en-fr.tst.en')

for model_class, tokenizer_class, pretrained_weights in MODELS:
  # Load pretrained model/tokenizer
  tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
  model = model_class.from_pretrained(pretrained_weights)

  process_fn = process_fn_(tokenizer)
  dataset = tf.data.Dataset.zip((tf.data.TextLineDataset('test.en'),tf.data.TextLineDataset('test.fr')))
  batch_size = 10  
  dataset = dataset.apply(training_pipeline(batch_size,
                    batch_type="examples",
                    batch_multiplier=1,
                    batch_size_multiple=1,
                    process_fn=process_fn,
                    length_bucket_width=1,
                    features_length_fn = lambda tokens: tf.shape(tokens)[0],
                    labels_length_fn = lambda tokens: tf.shape(tokens)[0],
                    maximum_features_length=50,
                    maximum_labels_length=50,
                    single_pass=False,
                    num_shards=1,
                    shard_index=0,
                    num_threads=None,
                    shuffle_buffer_size=None,
                    prefetch_buffer_size=200))
  dataset = dataset.map(lambda src, tgt: ({"input_ids":src, "langs":tf.tile(tf.Variable([["en"]]), tf.shape(src))},{"input_ids":tgt, "lang":tf.tile(tf.Variable([["fr"]]), tf.shape(tgt))}))
  for element in dataset:
    #print(model(element[0]))
    print(element)
    print(tokenizer.decode(element[0]["input_ids"].numpy()[0]))
    s1 = tf.reduce_mean(tf.reduce_mean(model(element[0])[0][0,:,:],0),0)
    s2 = tf.reduce_mean(tf.reduce_mean(model(element[1])[0][0,:,:],0),0)
    #print(s1)
    #print(s2)
    print(-tf.keras.losses.cosine_similarity(s1,s2))
    break

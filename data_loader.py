"""Dataset creation and transformations."""

import numpy as np
import tensorflow as tf
import sys
from transformers.tokenization_xlm import XLMTokenizer
import gzip
import os
from pathlib import Path

def _get_output_shapes(dataset):
  """Returns the outputs shapes of the dataset.

  Args:
    dataset: A ``tf.data.Dataset``.

  Returns:
    A nested structure of ``tf.TensorShape``
  """
  return tf.nest.map_structure(lambda spec: spec.shape, dataset.element_spec)

def get_dataset_size(dataset, batch_size=5000):
  """Get the dataset size.

  Args:
    dataset: A finite dataset.
    batch_size: The batch size to use or ``None`` to scan the dataset as-is.

  Returns:
    The dataset size.
  """
  if batch_size is not None:
    dataset = dataset.batch(batch_size)

  def _reduce_func(count, element):
    element = tf.nest.flatten(element)[0]
    batch_size = tf.shape(element)[0]
    return count + tf.cast(batch_size, count.dtype)

  return dataset.reduce(tf.constant(0, dtype=tf.int64), _reduce_func)

def filter_irregular_batches(multiple):
  """Transformation that filters out batches based on their size.

  Args:
    multiple: The divisor of the batch size.

  Returns:
    A ``tf.data.Dataset`` transformation.
  """
  if multiple == 1:
    return lambda dataset: dataset

  def _predicate(*x):
    flat = tf.nest.flatten(x)
    batch_size = tf.shape(flat[0])[0]
    return tf.equal(batch_size % multiple, 0)

  return lambda dataset: dataset.filter(_predicate)

def filter_examples_by_length(maximum_features_length=None,
                              maximum_labels_length=None,
                              features_length_fn=None,
                              labels_length_fn=None):
  
  if features_length_fn is None and labels_length_fn is None:
    return lambda dataset: dataset

  def _length_constraints(length, maximum_length):
    # Work with lists of lengths which correspond to the general multi source case.
    if not isinstance(length, list):
      length = [length]
    if not isinstance(maximum_length, list):
      maximum_length = [maximum_length]
    # Unset maximum lengths are set to None (i.e. no constraint).
    maximum_length += [None] * (len(length) - len(maximum_length))
    constraints = []
    for l, maxlen in zip(length, maximum_length):
      constraints.append(tf.greater(l, 0))
      if maxlen is not None:
        constraints.append(tf.less_equal(l, maxlen))
    return constraints

  def _predicate(features, labels):
    cond = []
    features_length = features_length_fn(features) if features_length_fn is not None else None
    labels_length = labels_length_fn(labels) if labels_length_fn is not None else None
    if features_length is not None:
      cond.extend(_length_constraints(features_length, maximum_features_length))
    if labels_length is not None:
      cond.extend(_length_constraints(labels_length, maximum_labels_length))
    return tf.reduce_all(cond)

  return lambda dataset: dataset.filter(_predicate)

def random_shard(shard_size, dataset_size):
  """Transformation that shards the dataset in a random order.

  Args:
    shard_size: The number of examples in each shard.
    dataset_size: The total number of examples in the dataset.

  Returns:
    A ``tf.data.Dataset`` transformation.
  """
  num_shards = -(-dataset_size // shard_size)  # Ceil division.
  offsets = np.linspace(0, dataset_size, num=num_shards, endpoint=False, dtype=np.int64)

  def _random_shard(dataset):
    sharded_dataset = tf.data.Dataset.from_tensor_slices(offsets)
    sharded_dataset = sharded_dataset.shuffle(num_shards)
    sharded_dataset = sharded_dataset.flat_map(
        lambda offset: dataset.skip(offset).take(shard_size))
    return sharded_dataset

  return _random_shard

def shuffle_dataset(buffer_size, shuffle_shards=True):  

  def _shuffle(dataset):
    sample_size = buffer_size
    if sample_size < 0 or shuffle_shards:
      dataset_size = get_dataset_size(dataset)
      tf.get_logger().info("Training on %d examples", dataset_size)
      if sample_size < 0:
        sample_size = dataset_size
      elif sample_size < dataset_size:
        dataset = dataset.apply(random_shard(sample_size, dataset_size))
    dataset = dataset.shuffle(sample_size)
    return dataset

  return _shuffle

def batch_dataset(batch_size, padded_shapes=None):
  """Transformation that batches a dataset.

  Args:
    batch_size: The batch size.
    padded_shapes: The padded shapes for this dataset. If ``None``, the shapes
      are automatically inferred from the dataset output shapes.

  Returns:
    A ``tf.data.Dataset`` transformation.

  See Also:
    :func:`opennmt.data.batch_sequence_dataset`
  """
  return lambda dataset: dataset.padded_batch(
      batch_size, padding_values=({"input_ids":2,"langs":0,"lengths":0},{"input_ids":2,"langs":1,"lengths":0}), padded_shapes=padded_shapes or _get_output_shapes(dataset))

def batch_sequence_dataset(batch_size,
                           batch_type="examples",
                           batch_multiplier=1,
                           batch_size_multiple=1,
                           length_bucket_width=None,
                           length_fn=None,
                           padded_shapes=None):
  """Transformation that batches a dataset of sequences.

  This implements an example-based and a token-based batching strategy
  with optional bucketing of sequences.

  Bucketing makes the batches contain sequences of similar lengths to optimize
  the training efficiency. For example, if :obj:`length_bucket_width` is 5,
  sequences will be organized by the following buckets:

  1 - 5 | 6 - 10 | 11 - 15 | ...

  Then when building the next batch, sequences will be selected from the same
  bucket.

  If the dataset has parallel elements (e.g. a parallel source and target
  dataset), the element is assigned to the bucket corresponding to the maximum
  length of all parallel elements.

  Args:
    batch_size: The batch size.
    batch_type: The training batching strategy to use: can be "examples" or
      "tokens".
    batch_multiplier: The batch size multiplier.
    batch_size_multiple: When :obj:`batch_type` is "tokens", ensure that the
      resulting batch size is a multiple of this value.
    length_bucket_width: The sequence length bucket width.
    length_fn: A function or list of functions (in case of a parallel dataset)
      that take features as argument and return the associated sequence length.
    padded_shapes: The padded shapes for this dataset. If ``None``, the shapes
      are automatically inferred from the dataset output shapes.

  Returns:
    A ``tf.data.Dataset`` transformation.

  Raises:
    ValueError: if :obj:`batch_type` is not one of "examples" or "tokens".

  See Also:
    :func:`opennmt.data.batch_dataset`
  """
  batch_size = batch_size * batch_multiplier

  def _get_bucket_id(features, length_fn):
    default_id = tf.constant(0, dtype=tf.int32)
    if length_fn is None:
      return default_id
    lengths = length_fn(features)
    if lengths is None:
      return default_id
    if not isinstance(lengths, list):
      lengths = [lengths]  # Fallback to the general case of parallel inputs.
    lengths = [length // length_bucket_width for length in lengths]
    return tf.reduce_max(lengths)

  def _key_func(*args):
    length_fns = length_fn
    if length_fns is None:
      length_fns = [None for _ in args]
    elif not isinstance(length_fns, (list, tuple)):
      length_fns = [length_fns]
    if len(length_fns) != len(args):
      raise ValueError("%d length functions were passed but this dataset contains "
                       "%d parallel elements" % (len(length_fns), len(args)))
    # Take the highest bucket id.
    bucket_id = tf.reduce_max([
        _get_bucket_id(features, length_fn)
        for features, length_fn in zip(args, length_fns)])
    return tf.cast(bucket_id, tf.int64)

  def _reduce_func(unused_key, dataset):
    return dataset.apply(batch_dataset(batch_size, padded_shapes=padded_shapes))

  def _window_size_func(key):
    if length_bucket_width > 1:
      key += 1  # For length_bucket_width == 1, key 0 is unassigned.
    size = batch_size // (key * length_bucket_width)
    required_multiple = batch_multiplier * batch_size_multiple
    if required_multiple > 1:
      size = size + required_multiple - size % required_multiple
    return tf.cast(tf.maximum(size, required_multiple), tf.int64)

  if length_bucket_width is None:
    return batch_dataset(batch_size, padded_shapes=padded_shapes)

  if batch_type == "examples":
    return tf.data.experimental.group_by_window(
        _key_func, _reduce_func, window_size=batch_size)
  elif batch_type == "tokens":
    return tf.data.experimental.group_by_window(
        _key_func, _reduce_func, window_size_func=_window_size_func)
  else:
    raise ValueError(
        "Invalid batch type: '{}'; should be 'examples' or 'tokens'".format(batch_type))

def process_fn_(tokenizer):
  @tf.autograph.experimental.do_not_convert
  def _tokenize_tensor(text, lang="en"):    
    def _python_wrapper(string_t):
      string = tf.compat.as_text(string_t.numpy())      
      tokens = tokenizer.encode(string,add_special_tokens=True)
      langs = [lang] * len(tokens)
      return tf.constant(tokens), tf.constant(langs)
    tokens, langs = tf.py_function(_python_wrapper, [text], (tf.int32, tf.int32))
    tokens.set_shape([None])
    langs.set_shape([None])
    return tokens, langs

  def process_fn(src,tgt):
    src_ids, langs_src = _tokenize_tensor(src, lang=0)
    tgt_ids, langs_tgt = _tokenize_tensor(tgt, lang=1)
    return ({"input_ids": src_ids, "langs": langs_src, "lengths": tf.shape(src_ids)[0]},
            {"input_ids": tgt_ids, "langs": langs_tgt, "lengths": tf.shape(tgt_ids)[0]})
  return process_fn

def training_pipeline(batch_size,
                      batch_type="examples",
                      batch_multiplier=1,
                      batch_size_multiple=1,
                      process_fn=None,
                      length_bucket_width=None,
                      features_length_fn=None,
                      labels_length_fn=None,
                      maximum_features_length=None,
                      maximum_labels_length=None,
                      single_pass=False,
                      num_shards=1,
                      shard_index=0,
                      num_threads=None,
                      prefetch_buffer_size=None):

  def _pipeline(dataset):
    if num_shards > 1:
      dataset = dataset.shard(num_shards, shard_index)
    
    if process_fn is not None:
      dataset = dataset.map(process_fn, num_parallel_calls=num_threads or 4)
    dataset = dataset.apply(filter_examples_by_length(
        maximum_features_length=maximum_features_length,
        maximum_labels_length=maximum_labels_length,
        features_length_fn=features_length_fn,
        labels_length_fn=labels_length_fn))
    dataset = dataset.apply(batch_sequence_dataset(
        batch_size,
        batch_type=batch_type,
        batch_multiplier=batch_multiplier,
        batch_size_multiple=batch_size_multiple,
        length_bucket_width=length_bucket_width,
        length_fn=[features_length_fn, labels_length_fn]))
    if not single_pass:
      dataset = dataset.repeat()
    dataset = dataset.prefetch(prefetch_buffer_size)
    return dataset

  return _pipeline

def inference_pipeline(batch_size,
                       process_fn=None,
                       length_bucket_width=None,
                       length_fn=None,
                       num_threads=None,
                       prefetch_buffer_size=None):

  def _inject_index(index, x):
    x["index"] = index
    return x

  def _pipeline(dataset):
    if process_fn is not None:
      dataset = dataset.map(process_fn, num_parallel_calls=num_threads)
    if length_bucket_width is not None and length_bucket_width > 0:
      if length_fn is None:
        raise ValueError("length_fn is required when reordering by length")
      if not isinstance(_get_output_shapes(dataset), dict):
        raise ValueError("Reordering by length expects dataset elements to be Python dicts")
      dataset = dataset.enumerate()
      dataset = dataset.map(_inject_index)
      dataset = dataset.apply(batch_sequence_dataset(
          batch_size,
          length_bucket_width=length_bucket_width,
          length_fn=length_fn))
    else:
      dataset = dataset.apply(batch_dataset(batch_size))
    dataset = dataset.prefetch(prefetch_buffer_size)
    return dataset

  return _pipeline

def function_on_next(dataset, as_numpy=False):
  """Decorator to run a ``tf.function`` on each dataset element.

  The motivation for this construct is to get the next element within the
  ``tf.function`` for increased efficiency.

  Args:
    dataset: The dataset to iterate.
    as_numpy: If ``True``, call `.numpy()` on each output tensors.

  Returns:
    A function decorator. The decorated function is transformed into a callable
    that returns a generator over its outputs.
  """

  def decorator(func):
    def _fun():
      iterator = iter(dataset)

      @tf.function
      def _tf_fun():
        return func(lambda: next(iterator))

      while True:
        try:
          outputs = _tf_fun()
          if as_numpy:
            outputs = tf.nest.map_structure(lambda x: x.numpy(), outputs)
          yield outputs
        except tf.errors.OutOfRangeError:
          break

    return _fun

  return decorator

class Dataset() :

  def __init__(self,                             
              filepath,                
              training_data_save_path,
              seq_size, 
              max_sents, 
              do_shuffle, 
              do_skip_empty,
              procedure="training",
              model_name_or_path = 'xlm-mlm-enfr-1024',
              tokenizer_class = XLMTokenizer,
              tokenizer_cache_dir = "tokenizer"):

    if filepath is None:
      sys.stderr.write("error: give some filepath")
      sys.exit(1)
    
    Path(training_data_save_path).mkdir(parents=True, exist_ok=True)
    self.files = filepath.split(",")
    self.seq_size = seq_size
    self.max_sents = max_sents
    self.do_shuffle = do_shuffle
    self.do_skip_empty = do_skip_empty
    self.annotated = False
    self.data = []
    ### length of the data set to be used (not necessarily the whole set)
    self.length = 0
    self.tokenizer = tokenizer_class.from_pretrained(model_name_or_path, cache_dir=tokenizer_cache_dir if tokenizer_cache_dir else None)
    assert len(self.files) == 3 or len(self.files) == 2    
    self.false_tgt = os.path.join(training_data_save_path,"%s.tgt.false"%procedure)
    self.src = os.path.join(training_data_save_path,"%s.src"%procedure)
    self.tgt = os.path.join(training_data_save_path,"%s.tgt"%procedure)

  def get_tokenizer(self):
    return self.tokenizer

  def shuffle(self):
    with open(self.files[0],"r") as f:      
      line_read_src = f.read() 
      line_read_src = [l.strip() for l in line_read_src]
    with open(self.files[1],"r") as f:
      line_read_tgt = f.read()
      line_read_tgt = [l.strip() for l in line_read_tgt]

    self.dataset_size = len(line_read_src)
    inds = np.arange(self.dataset_size)
    from random import shuffle
    shuffle(inds)
    with open(self.src,"w") as f_write_src:
      for id in inds:
        print(line_read_src[id], file=f_write_src)

    with open(self.tgt,"w") as f_write_tgt:
      for id in inds:
        print(line_read_tgt[id], file=f_write_tgt)

    with open(self.false_tgt,"w") as f_write_false_tgt:
      for id in inds:
        false_tgt_id = (id + np.random.choice(self.dataset_size,1)[0])%self.dataset_size
        print(line_read_tgt[false_tgt_id], file=f_write_false_tgt)

  def copy(self):

    with open(self.files[0],"r") as f:      
      line_read_src = f.read() 
      line_read_src = [l.strip() for l in line_read_src]
    with open(self.files[1],"r") as f:
      line_read_tgt = f.read()
      line_read_tgt = [l.strip() for l in line_read_tgt]
    self.dataset_size = len(line_read_src)
    inds = np.arange(self.dataset_size)
    with open(self.src,"w") as f_write_src:
      with open(self.tgt,"w") as f_write_tgt:
        with open(self.false_tgt,"w") as f_write_false_tgt:
          for id in inds:
            print(line_read_src[id], file=f_write_src)
            print(line_read_tgt[id], file=f_write_tgt)
            false_tgt_id = (id + np.random.choice(self.dataset_size,1)[0])%self.dataset_size
            print(line_read_tgt[false_tgt_id], file=f_write_false_tgt)

  def create_one_epoch(self, do_shuffle=True, mode="p"):
    print("Creating training data files")
    if do_shuffle:
      self.shuffle()
    else:
      self.copy()
    print("finished creating training data files")
    process_fn = process_fn_(self.tokenizer)
    if mode =="p":
      dataset = tf.data.Dataset.zip((tf.data.TextLineDataset(self.src),tf.data.TextLineDataset(self.tgt)))
    elif mode == "u":
      dataset = tf.data.Dataset.zip((tf.data.TextLineDataset(self.src),tf.data.TextLineDataset(self.false_tgt)))
    os.remove(self.src)
    os.remove(self.tgt)
    os.remove(self.false_tgt)
    print("finish remove training data files")
    batch_size = self.max_sents
    dataset = dataset.apply(training_pipeline(batch_size,
                      batch_type="examples",
                      batch_multiplier=1,
                      batch_size_multiple=1,
                      process_fn=process_fn,
                      length_bucket_width=None,
                      features_length_fn = lambda src: tf.shape(src["input_ids"])[0],
                      labels_length_fn = lambda tgt: tf.shape(tgt["input_ids"])[0],
                      maximum_features_length=100,
                      maximum_labels_length=100,
                      single_pass=True,
                      num_shards=1,
                      shard_index=0,
                      num_threads=None,
                      prefetch_buffer_size=200))

    return dataset
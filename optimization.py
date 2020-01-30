import tensorflow as tf

class ScheduleWrapper(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Wrapper to augment a learning rate scheduler behavior."""

  def __init__(self,
               schedule,
               step_start=0,
               step_duration=1,
               minimum_learning_rate=0):
    """Initializes the decay function.

    Args:
      schedule: A ``tf.keras.optimizers.schedules.LearningRateSchedule``.
      step_duration: The number of training steps that make 1 decay step.
      start_step: Start decay after this many steps.
      minimum_learning_rate: Do not decay past this learning rate value.

    See Also:
      :class:`opennmt.schedules.make_learning_rate_schedule`
    """
    self.schedule = schedule
    self.step_start = step_start
    self.step_duration = step_duration
    self.minimum_learning_rate = minimum_learning_rate

  def __call__(self, step):
    # Map the training step to a decay step.
    step = tf.maximum(step - self.step_start, 0)
    step //= self.step_duration
    learning_rate = self.schedule(step)
    return tf.maximum(learning_rate, self.minimum_learning_rate)
class NoamDecay(tf.keras.optimizers.schedules.LearningRateSchedule):

  """Defines the decay function described in https://arxiv.org/abs/1706.03762."""

  def __init__(self, scale, model_dim, warmup_steps):
    """Initializes the decay function.

    Args:
      scale: The scale constant.
      model_dim: The model dimension.
      warmup_steps: The number of warmup steps.
    """
    self.scale = tf.cast(scale, tf.float32)
    self.model_dim = tf.cast(model_dim, tf.float32)
    self.warmup_steps = tf.cast(warmup_steps, tf.float32)

  def __call__(self, step):
    step = tf.cast(step + 1, tf.float32)
    return (self.scale
            * tf.pow(self.model_dim, -0.5)
            * tf.minimum(tf.pow(step, -0.5), step * tf.pow(self.warmup_steps, -1.5)))

class GradientAccumulator(object):
  """Distribution strategies-aware gradient accumulation utility."""

  def __init__(self):
    """Initializes the accumulator."""
    self._gradients = []
    self._accum_steps = tf.Variable(
        initial_value=0,
        dtype=tf.int64,
        trainable=False,
        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

  @property
  def step(self):
    """Number of accumulated steps."""
    return self._accum_steps.value()

  @property
  def gradients(self):
    """The accumulated gradients."""
    return list(gradient.value() for gradient in self._get_replica_gradients())

  def __call__(self, gradients):
    """Accumulates :obj:`gradients`."""
    if not self._gradients:
      self._gradients.extend([
          tf.Variable(tf.zeros_like(gradient), trainable=False)
          for gradient in gradients])
    if len(gradients) != len(self._gradients):
      raise ValueError("Expected %s gradients, but got %d" % (
          len(self._gradients), len(gradients)))

    for accum_gradient, gradient in zip(self._get_replica_gradients(), gradients):
      accum_gradient.assign_add(gradient)

    self._accum_steps.assign_add(1)

  def reset(self):
    """Resets the accumulated gradients."""
    if self._gradients:
      self._accum_steps.assign(0)
      for gradient in self._get_replica_gradients():
        gradient.assign(tf.zeros_like(gradient))

  def _get_replica_gradients(self):
    if tf.distribute.has_strategy():
      # In a replica context, we want to accumulate gradients on each replica
      # without synchronization, so we directly assign the value of the
      # current replica.
      replica_context = tf.distribute.get_replica_context()
      if replica_context is None:
        return self._gradients
      return (
          gradient.device_map.select_for_current_replica(gradient.values, replica_context)
          for gradient in self._gradients)
    else:
      return self._gradients


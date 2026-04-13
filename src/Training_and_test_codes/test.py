import tensorflow as tf

print("✅ TF version:", tf.__version__)
print("✅ Built with CUDA:", tf.test.is_built_with_cuda())
print("✅ Physical GPUs:", tf.config.list_physical_devices('GPU'))
print("✅ Logical GPUs:", tf.config.list_logical_devices('GPU'))

# Force a simple GPU operation
tf.debugging.set_log_device_placement(True)

a = tf.constant([[1.0, 2.0]])
b = tf.constant([[3.0], [4.0]])
c = tf.matmul(a, b)
print("✅ Matmul result:", c.numpy())

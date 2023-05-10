import tensorflow as tf

print(tf.version)

t = tf.zeros([5, 5, 5, 5])
# print(t)
t = tf.reshape(t, [125, -1])
print(t)
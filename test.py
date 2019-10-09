import numpy as np
import tensorflow as tf


a=tf.train.range_input_producer(5).dequeue()
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
sess.run(a)
print(a)
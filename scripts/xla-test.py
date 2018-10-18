# XLA compilation controlled by "compile_ops" option
# compile_ops=False: 4.39 sec
# compile_ops=True: 0.90 sec
# Note this works for both CPU and GPU


import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import tensorflow as tf


from tensorflow.contrib.compiler import jit
tf.reset_default_graph()
jit_scope = jit.experimental_jit_scope
with jit_scope(compile_ops=True):
    N = 500*1000*1000
    x = tf.Variable(tf.random_uniform(shape=(N,)))
    y = 0.1*x*x*x*x*x-0.5*x*x*x*x+.25*x*x*x+.75*x*x-1.5*x-2
    y0 = y[0]
    
import time
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(y.op)
start_time = time.time()
print(sess.run(y0))
end_time = time.time()
print("%.2f sec"%(end_time-start_time))

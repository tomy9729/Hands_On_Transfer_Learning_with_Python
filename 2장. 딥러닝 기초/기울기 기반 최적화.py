import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt

tf.disable_v2_behavior()
#x,y = (2,2) 경사 하강 초기화
x = tf.Variable(2, name='x', dtype=tf.float32)
y = tf.Variable(2, name='y', dtype=tf.float32)

temperature = 50 - tf.square(y) - 2*tf.square(x)

#Gradient Descent Optimizer 초기화
optimizer = tf.train.GradientDescentOptimizer(0.1) # 학습율 0.1
train = optimizer.minimize(temperature)
grad = tf.gradients(temperature, [x,y]) #기울기 벡터를 계산해보자

init = tf.global_variables_initializer()
with tf.Session() as session : 
  session.run(init)
  print("Starting at coordinate x={}, y={} and temperature there is. {}".format(session.run(x), session.run(y),session.run(temperature)))
  grad_norms = []
  for step in range(10) : 
    session.run(train)
    g = session.run(grad)
    print("step({}) x={}, y={}, T={}, Gradient={}".format(step,session.run(x),session.run(y),session.run(temperature),g))
    grad_norms.append(np.linalg.norm(g))

plt.plot(grad_norms)

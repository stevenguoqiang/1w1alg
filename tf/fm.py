import numpy as np
import tensorflow as tf

x_data = np.matrix([[1,0,0,1,0,0,0,0.3,0.3,0.3,0]])
y_data = np.array([5])
y_data.shape += (1,)

n,p = x_data.shape
k =5

X = tf.placeholder('float', shape=[n,p])
y = tf.placeholder('float', shape=[n,1])

w0 = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.zeros[p])

V = tf.Variable(tf.random_normal([k,p], stddev=0.01))
y_hat = tf.Variable(tf.zeros[n,1])


linear_terms = tf.add(w0, tf.readuce_sum(tf.multiply(W,X), 1, keep_dims=True))
interactions = (tf.multiply(0.5, tf.reduce_sum(
    tf.sub(tf.pow(tf.matmul(X,tf.transpose(V)),2),
           tf.matmul(tf.pow(X,2), tf.transpose(tf.pow(V,2)))),
    1,keep_dims=True)))
y_hat = tf.add(linear_terms, interactions)

lambda_w = tf.constant(0.001, name='lmd_w')
lambda_v = tf.constant(0.001, name='lmd_v')

l2_norm = (tf.reduce_sum(
    tf.add(
        tf.multiply(lambda_w, tf.pow(W,2)),
        tf.multiply(lambda_v, tf.pow(V,2)))))

error = tf.reduce_mean(tf.square(tf.sub(y,y_hat)))
loss = tf.add(error, l2_norm)

eta = tf.constant(0.1)
optimizer = tf.train.AdagradOptimizer(eta).minimize(loss)

N_EPOCHS = 1000
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(N_EPOCHS):
        indices = np.arange(n)
        np.random.shuffle(indices)
        x_data, y_data = x_data[indices], y_data[indices]
        sess.run(optimizer, feed_dict={X:x_data, y:y_data})
    print("MSE: ", sess.run(error, feed_dict={X:x_data, y_y_data})
    )



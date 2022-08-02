import numpy as np
import autodiff as ad
import matplotlib.pyplot as plt

x = ad.Variable(name = "x")
w = ad.Variable(name = "w")
b = ad.Variable(name = "b")
labels = ad.Variable(name = "lables")


# TODO: 使用给定的Ops, 实现sigmoid函数
def sigmoid(x):
	# rst = ad.div_constby_op(ad.add_byconst_op(ad.exp_op(ad.sub_constby_op(0, x)), 1), 1)
	rst = 1.0 / (1.0 + ad.exp_op(-1.0 * x))
	return rst

# TODO: 使用给定的Ops, 实现逻辑回归的BCE损失函数
def bce_loss(xs, labels):
	# loss = ad.sub_constby_op(
	# 		ad.add_op(
	# 			ad.mul_op(labels, ad.log_op(xs)),
	# 			ad.mul_op(ad.sub_constby_op(labels, 1), ad.log_op(ad.sub_constby_op(xs, 1)))),
	# 		0)
	loss = -1.0 * ad.reduce_sum_op(labels * ad.log_op(xs) + (1.0 - labels) * ad.log_op(1.0 - xs), axis=1)
	return loss

p = sigmoid(ad.matmul_op(w, x))
loss = bce_loss(p, labels) 

grad_y_w, = ad.gradients(loss, [w])


num_features = 2
num_points = 200
num_iterations = 1000
learning_rate = 0.01

# The dummy dataset consists of two classes.
# The classes are modelled as a random normal variables with different means.

class_1 = np.random.normal(2, 0.1, (100, num_features))
class_2 = np.random.normal(4, 0.1, (100, num_features))
x_val = np.concatenate((class_1, class_2), axis = 0).T

x_val = np.concatenate((x_val, np.ones((1, num_points))), axis = 0)
w_val = np.random.normal(size = (1, num_features + 1))


labels_val = np.concatenate((np.zeros((class_1.shape[0], 1)), np.ones((class_2.shape[0], 1))), axis=0).T
executor = ad.Executor([loss, grad_y_w])

for i in range(100000):
	# evaluate the graph
	loss_val, grad_y_w_val =  executor.run(feed_dict={x:x_val, w:w_val, labels:labels_val})
	# TODO: update the parameters using SGD
	w_val = w_val - learning_rate * grad_y_w_val
	if i % 1000 == 0:
		print(loss_val)
import theano
import numpy as np
import matplotlib.pyplot as plt

from theano import tensor as T

# init
x1 = T.scalar()
w1 = T.scalar()
w0 = T.scalar()
z1 = w1 * x1 + w0

# compile
net_input = theano.function(inputs=[w1, x1, w0], outputs=z1)

# execute
print('Net input: %.2f' % net_input(2.0, 1.0, 0.5))

# arrays
x = T.dmatrix(name='x') # fmatrix for 32-bit
x_sum = T.sum(x, axis=0)

calc_sum = theano.function(inputs=[x], outputs=x_sum)

ary = [[1, 2, 3], [1, 2, 3]]
print('Column sum:', calc_sum(ary))

ary = np.array([[1, 2, 3], [1, 2, 3]], dtype=theano.config.floatX)
print('Column sum:', calc_sum(ary))

# shared vars
x = T.dmatrix(name='x')
w = theano.shared(np.asarray([[0.0, 0.0, 0.0]], dtype=theano.config.floatX))

z = x.dot(w.T)
update = [[w, w + 1.0]]

net_input = theano.function(inputs=[x], updates=update, outputs=z)

data = np.array([[1, 2, 3]], dtype=theano.config.floatX)
print('Shared vals')
for i in range(5):
    print('z%d:' % i, net_input(data))

# given vars aka givens
data = np.array([[1, 2, 3]], dtype=theano.config.floatX)
x = T.dmatrix(name='x')
w = theano.shared(np.asarray([[0.0, 0.0, 0.0]], dtype=theano.config.floatX))

z = x.dot(w.T)
update = [[w, w + 1.0]]

net_input = theano.function(inputs=[], updates=update, givens={x: data}, outputs=z)
print('Givens')
for i in range(5):
    print('z%d:' % i, net_input())

# linear reg demo

x_train = np.asarray([[0.0], [1.0],
                      [2.0], [3.0],
                      [4.0], [5.0],
                      [6.0], [7.0],
                      [8.0], [9.0]],
                     dtype=theano.config.floatX)

y_train = np.asarray([1.0, 1.3,
                      3.1, 2.0,
                      5.0, 6.3,
                      6.6, 7.4,
                      8.0, 9.0],
                     dtype=theano.config.floatX)


def train_linreg(x_train, y_train, eta, epochs):
    costs = []
    eta0 = T.dscalar('eta0')
    y = T.dvector(name='y')
    x = T.dmatrix(name='x')
    w = theano.shared(np.zeros(shape = (x_train.shape[1] + 1),
                                     dtype=theano.config.floatX), name='w')

    # calc host
    net_input = T.dot(x, w[1:]) + w[0]
    errors = y - net_input
    cost = T.sum(T.pow(errors, 2))

    # perform gradient update
    gradient = T.grad(cost, wrt=w)
    update = [(w, w - eta0 * gradient)]

    # compile model
    train = theano.function(inputs=[eta0], outputs=cost,
                                  updates=update,
                                  givens={x: x_train, y: y_train})
    for _ in range(epochs):
        costs.append(train(eta))

    return costs, w


def predict_linreg(x, w):
    xt = T.matrix(name='x')
    net_input = T.dot(xt, w[1:] + w[0])
    predict = theano.function(inputs=[xt], givens={w: w}, outputs=net_input)

    return predict(x)


costs, w = train_linreg(x_train, y_train, eta=0.001, epochs=10)
plt.plot(range(1, len(costs)+1), costs)
plt.tight_layout()
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()

plt.scatter(x_train, y_train, marker='s', s=50)
plt.plot(range(x_train.shape[0]), predict_linreg(x_train, w),
         color='gray', marker='o', markersize=4, linewidth=3)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

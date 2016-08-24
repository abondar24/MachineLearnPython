import numpy as np
import matplotlib.pyplot as plt

scores = [3.0, 1.0, 0.2]


# clasifier which makes prediction on kx+b func
def softmax(x):
    """compute soft max values for x"""
    return np.exp(x)/np.sum(np.exp(x), axis=0)


x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2*np.ones_like(x)])

print(softmax(scores))
plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()

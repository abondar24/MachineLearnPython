import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets.samples_generator import make_circles


def bucket_mean(data, bucket_ids, num_buckets):
    total = tf.unsorted_segment_sum(data, bucket_ids, num_buckets)
    count = tf.unsorted_segment_sum(tf.ones_like(data), bucket_ids, num_buckets)
    return total / count


DATA_TYPE = 'blobs'
N = 200
# num of clusters
if DATA_TYPE == 'circle':
    K = 2
else:
    K = 4

# max number of iterations, if conditions aren't met
MAX_ITERS = 1000

start = time.time()

centers = [(-2, -2), (-2, 1.5), (1.5, -2), (2, 1.5)]
if DATA_TYPE == 'circle':
    data, feautres = make_circles(n_samples=200, shuffle=True, noise=0.01, factor=0.4)
else:
    data, feautres = make_blobs(n_samples=200, centers=centers, cluster_std=0.8,
                                shuffle=False, random_state=42)

fig, ax = plt.subplots()
ax.scatter(np.asarray(centers).transpose()[0], np.asarray(centers).transpose()[1],
           marker='o', s=250)
plt.show()

fig, ax = plt.subplots()
if DATA_TYPE == 'blobs':
    ax.scatter(np.asarray(centers).transpose()[0], np.asarray(centers).transpose()[1],
               marker='o', s=250)
    ax.scatter(data.transpose()[0], data.transpose()[1], marker='o', s=100, c=feautres)
    plt.show()

points = tf.Variable(data)
cluster_assigments = tf.Variable(tf.zeros([N], dtype=tf.int64))
centroids = tf.Variable(tf.slice(points.initialized_value(), [0, 0], [K, 2]))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

sess.run(centroids)
# loss function

# N copies of each centroid
rep_centroids = tf.reshape(tf.tile(centroids, [N, 1]), [N, K, 2])

# K copies of each point
rep_points = tf.reshape(tf.tile(points, [1, K]), [N, K, 2])

# NxK copies of every point
sum_squares = tf.reduce_sum(tf.square(rep_points - rep_centroids),
                            reduction_indices=2)

# index of centroid assigned to each point
best_centroids = tf.argmin(sum_squares, 1)

# stop condition
did_assigments_change = tf.reduce_any(tf.not_equal(best_centroids, cluster_assigments))

means = bucket_mean(points, best_centroids, K)

# calculate whether updated needed or not
with tf.control_dependencies([did_assigments_change]):
    do_updates = tf.group(centroids.assign(means),
                          cluster_assigments.assign(best_centroids))

changed = True
iters = 0

fig, ax = plt.subplots()
if DATA_TYPE == 'blobs':
    colour_indexes = [2, 1, 4, 3]
else:
    colour_indexes = [2, 1]

while changed and iters < MAX_ITERS:
    fig, ax = plt.subplots()
    iters += 1
    [changed, _] = sess.run([did_assigments_change, do_updates])
    [centers, assignments] = sess.run([centroids, cluster_assigments])
    ax.scatter(sess.run(points).transpose()[0], sess.run(points).transpose()[1],
               marker='o', s=200, c=assignments)
    ax.scatter(centers[:, 0], centers[:, 1], marker='^', s=550, c=colour_indexes)
    ax.set_title('Iteration ' + str(iters))
    plt.savefig("kmeans" + str(iters) + ".png")

ax.scatter(sess.run(points).transpose()[0], sess.run(points).transpose()[1],
           marker='o', s=200, c=assignments)
end = time.time()

print("Found in %.2f seconds" % (end - start)), iters, "iterations"
print("Centroids:")
print(centers)
print("Cluster assignments:", assignments)


import tensorflow as tf

filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("/home/abondar/test.gif"))

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

image = tf.image.decode_gif(value)

coord = tf.train.Coordinator()


def normalize_and_encode(img_tensor):
    image_dimensions = tf.shape(img_tensor.eval()[0]).eval()
    return tf.image.encode_jpeg(tf.reshape(tf.cast(img_tensor, tf.uint8), image_dimensions))


with tf.Session() as sess:
    max_file = open("max_pool.jpeg", "wb+")
    avg_file = open("avg_pool.jpeg", "wb+")
    tf.global_variables_initializer().run()

    threads = tf.train.start_queue_runners(coord=coord)

    image_tensor = tf.image.rgb_to_grayscale(sess.run(image))
    maxed_tensor = tf.nn.avg_pool(tf.cast(image_tensor, tf.float32),
                                  [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
    averaged_tensor = tf.nn.avg_pool(tf.cast(image_tensor, tf.float32),
                                     [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

    max_file.write(normalize_and_encode(maxed_tensor).eval())
    avg_file.write(normalize_and_encode(averaged_tensor).eval())

    coord.request_stop()
    max_file.close()
    avg_file.close()
coord.join(threads)

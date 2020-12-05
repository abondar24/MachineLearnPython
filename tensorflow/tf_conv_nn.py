import tensorflow as tf

filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("../data/test.gif"))
reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)
image = tf.image.decode_gif(value)

# define the kernel params
kernel = tf.constant([
    [[[-1.]], [[-1.]], [[-1.]]],
    [[[-1.]], [[8.]], [[-1.]]],
    [[[-1.]], [[-1.]], [[-1.]]]
])

# define the train coordinator
coord = tf.train.Coordinator()

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    threads = tf.train.start_queue_runners(coord=coord)

    # get image
    image_tensor = tf.image.rgb_to_grayscale(sess.run([image])[0])

    # apply convolution
    imagen_conv_tensor = tf.nn.conv2d(tf.cast(image_tensor, tf.float32), kernel,
                                      [1, 1, 1, 1], "SAME")

    # save conv option
    file = open("blur.jpg", "wb+")

    # cast tp uint8 prev scalation , because conv could alter the scale of the final image
    out = tf.image.encode_jpeg(tf.reshape(tf.cast(imagen_conv_tensor /
                                                  tf.reduce_max(imagen_conv_tensor) * 255., tf.uint8),
                                          tf.shape(imagen_conv_tensor.eval()[0]).eval()))
    file.write(out.eval())
    file.close()
    coord.request_stop()
coord.join(threads)

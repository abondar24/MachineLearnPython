from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
import os
import pickle

from tf_lstm_char_seq_utils import TextLoader
from tf_lstm_char_seq_model import Model
from six import text_type


# gen args class
class arguments:
    save_dir = 'char_seq'
    n = 1000
    prime = 'x:1\n'
    sample = 1


def main():
    args = arguments()
    sample(args)


def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        # load config from file
        saved_args = pickle.load(f)

    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        # load vocabulary
        chars, vocab = pickle.load(f)

    # rebuild the model
    model = Model(saved_args, True)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.all_variables())

        # retrieve the chkpoint
        ckpt = tf.train.get_checkpoint_state(args.save_dir)

        if ckpt and ckpt.model_checkpoint_path:

            # restore the model
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(model.sample(sess, chars, vocab, args.n, args.prime, args.sample))


# exec the model, generating a n char sequence starting with the prime seq
if __name__ == '__main__':
    main()

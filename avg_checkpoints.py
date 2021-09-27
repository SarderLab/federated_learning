"""Script to average values of variables in a list of checkpoint files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import os
import numpy as np
import six
from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow.compat.v1 as tf

def avg_checkpoints(input_checkpoints="", num_last_checkpoints=0, prefix="", output_path="averaged.ckpt", global_step_start=0):
    """Script to average values of variables in a list of checkpoint files."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # do not use GPU

    # flags.DEFINE_string("checkpoints", "",
    #                     "Comma-separated list of checkpoints to average.")
    # flags.DEFINE_integer("num_last_checkpoints", 0,
    #                      "Averages the last N saved checkpoints."
    #                      " If the checkpoints flag is set, this is ignored.")
    # flags.DEFINE_string("prefix", "",
    #                     "Prefix (e.g., directory) to append to each checkpoint.")
    # flags.DEFINE_string("output_path", "/tmp/averaged.ckpt",
    #                     "Path to output the averaged checkpoint to.")

    def checkpoint_exists(path):
        return (tf.gfile.Exists(path) or tf.gfile.Exists(path + ".meta") or
                tf.gfile.Exists(path + ".index"))


    if input_checkpoints:
        # Get the checkpoints list from flags and run some basic checks.
        checkpoints = [c.strip() for c in input_checkpoints]
        checkpoints = [c for c in checkpoints if c]
        if not checkpoints:
            raise ValueError("No checkpoints provided for averaging.")
        if prefix:
            checkpoints = [prefix + c for c in checkpoints]

    else:
        assert num_last_checkpoints >= 1, "Must average at least one model"
        assert prefix, ("Prefix must be provided when averaging last"
                                                    " N checkpoints")
        checkpoint_state = tf.train.get_checkpoint_state(
                os.path.dirname(prefix))
        # Checkpoints are ordered from oldest to newest.
        checkpoints = checkpoint_state.all_model_checkpoint_paths[
                -num_last_checkpoints:]

    checkpoints = [c for c in checkpoints if checkpoint_exists(c)]
    if not checkpoints:
        if input_checkpoints:
            raise ValueError(
                    "None of the provided checkpoints exist. %s" % input_checkpoints)
        else:
            raise ValueError("Could not find checkpoints at %s" %
                                             os.path.dirname(prefix))

    # Read variables from all checkpoints and average them.
    tf.logging.info("Reading variables and averaging checkpoints:")
    for c in checkpoints:
        tf.logging.info("%s ", c)
    var_list = tf.train.list_variables(checkpoints[0])
    var_values, var_dtypes = {}, {}
    for (name, shape) in var_list:
        if not name.startswith("global_step"):
            var_values[name] = np.zeros(shape)
    for checkpoint in checkpoints:
        reader = tf.train.load_checkpoint(checkpoint)
        for name in var_values:
            tensor = reader.get_tensor(name)
            var_dtypes[name] = tensor.dtype
            var_values[name] += tensor
        tf.logging.info("Read from checkpoint %s", checkpoint)
        print('\tRead from checkpoint {}'.format(checkpoint))
    for name in var_values:    # Average.
        var_values[name] /= len(checkpoints)

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        tf_vars = [
                tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[v])
                for v in var_values
        ]
    placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
    assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
    global_step = tf.Variable(
            global_step_start, name="global_step", trainable=False, dtype=tf.int64)
    saver = tf.train.Saver(tf.all_variables())

    # Build a model consisting only of variables, set them to the average values.
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total = len(placeholders)
        for iter, (p, assign_op, (name, value)) in enumerate(zip(placeholders, assign_ops,
                    six.iteritems(var_values))):
            sess.run(assign_op, {p: value})
            print('\tassigning parameter [{} of {}]: {}'.format(iter, total, name), end='\r')
        print('\n\n')
        # Use the built saver to save the averaged checkpoint.
        saver.save(sess, output_path, global_step=global_step)

    tf.logging.info("Averaged checkpoints saved in %s", output_path)

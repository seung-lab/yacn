# plan: We need to give the autoencoder some more memory,
# and give it access to the higher level features already computed.

# implement hard attention

# We should have local feature extraction and recurrent tracing
# You should think of the model as being inactive most of the time.
# We need to implement hard attention in order to move to the next window
# Memory is is spatially indexed

#We need an adversary which checks the output to see if it is distinguishable from the output on test data.


from __future__ import print_function
import numpy as np
import os
from datetime import datetime
import time
import math
import itertools
import threading
import pprint
from convkernels import *
from activations import *

import tensorflow as tf
from tensorflow.python.client import timeline

from utils import (identity_matrix,
                   image_summary, categorical,
                   conditional_map, covariance, image_slice_summary,
                   get_pair, bounded_cross_entropy, block_cat, batch_matmul,
                   batch_transpose, pad_shape, vec_cat)
from dataset import (gen, Dataset, SNEMI3D_TRAIN_DIR, SNEMI3D_TEST_DIR,
                     alternating_iterator)
from experiments import save_experiment, repo_root


class Model():

    def __init__(self, patch_size, offsets, mask_boundary,
                 mask_unknown, nvec_labels, maxn,
                 devices):

        self.summaries = []
        self.devices = devices
        self.patch_size = patch_size
        self.offsets = offsets
        self.mask_boundary = mask_boundary
        self.mask_unknown = mask_unknown
        self.maxn = maxn
        self.nvec_labels = nvec_labels
        self.propagator = identity_matrix(nvec_labels)
        # tf.matrix_inverse(self.propagator)
        self.inv_propagator = identity_matrix(nvec_labels)

        config = tf.ConfigProto(
            # gpu_options = tf.GPUOptions(allow_growth=True),
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9),
            # log_device_placement=True,
            # intra_op_parallelism_threads=1,
            # inter_op_parallelism_threads=1,
        )
        self.sess = tf.Session(config=config)
        self.run_metadata = tf.RunMetadata()

        inpt_placeholder = tf.placeholder(tf.float32, shape=patch_size + (1,))
        human_labels_placeholder = tf.placeholder(
            tf.int32, shape=patch_size + (1,))
        iteration_type_placeholder = tf.placeholder(tf.int32, shape=())
        seed_placeholder = tf.placeholder(tf.float32, shape=patch_size + (1,))

        self.q = tf.FIFOQueue(5, [tf.float32, tf.int32, tf.int32], shapes=[
                              patch_size + (1,), patch_size + (1,), ()])
        self.enqueue_op = self.q.enqueue(
            [inpt_placeholder,
             human_labels_placeholder,
             iteration_type_placeholder])

        inpt, human_labels, iteration_type = self.q.dequeue()
        self.inpt = inpt
        self.human_labels = human_labels
        self.iteration_type = iteration_type

        with tf.name_scope('params'):
            self.step = tf.Variable(0)
            strides = [(2, 2) for i in xrange(5)]
            sizes = [(4, 4, 1), (4, 4, 2), (4, 4, 4), (4, 4, 8), (4, 4, 16)]

            initial_schemas = [
                        FeatureSchema(1,0),
                        FeatureSchema(24,1),
                        FeatureSchema(28,2),
                        FeatureSchema(32,3),
                        FeatureSchema(48,4),
                        FeatureSchema(64,5)]
            second_schemas = [
                        FeatureSchema(nvec_labels,0),
                        FeatureSchema(24,1),
                        FeatureSchema(28,2),
                        FeatureSchema(32,3),
                        FeatureSchema(48,4),
                        FeatureSchema(64,5)]
            connection_schemas = [
                        Connection3dSchema(size=(4,4,1),strides=(2,2)),
                        Connection3dSchema(size=(4,4,2),strides=(2,2)),
                        Connection3dSchema(size=(4,4,4),strides=(2,2)),
                        Connection3dSchema(size=(4,4,8),strides=(2,2)),
                        Connection3dSchema(size=(4,4,16),strides=(2,2))]

            feature_schemas = [
                        VectorLabelSchema(12,0,nvec_labels),
                        LabelledFeatureSchema(24,1,nvec_labels),
                        LabelledFeatureSchema(28,2,nvec_labels),
                        LabelledFeatureSchema(32,3,nvec_labels),
                        LabelledFeatureSchema(48,4,nvec_labels),
                        LabelledFeatureSchema(64,5,nvec_labels)]

            initial_activations = [
                lambda x: x,
                tf.nn.elu,
                tf.nn.elu,
                tf.nn.elu,
                tf.nn.elu,
                tf.nn.elu]
            def lift(f):
                s=SymmetricTanh(reduction_index=1)
                return lambda x: FocusedMap(f(x.features),s(x.labels))
            def lift2(f):
                s=SymmetricTanh(reduction_index=1)
                return lambda x: VectorLabelMap(f(x.features),s(x.labels))
            activations = [
                tf.nn.elu,
                tf.nn.elu,
                tf.nn.elu,
                tf.nn.elu,
                tf.nn.elu,
                tf.nn.elu
                ]
            second_activations = [
                lift2(SymmetricTanh()),
                lift(tf.nn.elu),
                lift(tf.nn.elu),
                lift(tf.nn.elu),
                lift(tf.nn.elu),
                lift(tf.nn.elu)
                ]
            initial = MultiscaleUpConv3d(
                    feature_schemas = initial_schemas,
                    connection_schemas = connection_schemas,
                activations=initial_activations)
            it1 = MultiscaleConv3d(initial_schemas, second_schemas, connection_schemas, connection_schemas, activations)
            it2 = MultiscaleConv3d(second_schemas, second_schemas, connection_schemas, connection_schemas, activations)
            it3 = MultiscaleConv3d(second_schemas, second_schemas, connection_schemas, connection_schemas, activations)
            convert = MultiscaleCast(second_schemas, feature_schemas)
            it4 = MultiscaleConv3d(feature_schemas, feature_schemas, connection_schemas, connection_schemas, second_activations)
            it5 = MultiscaleConv3d(feature_schemas, feature_schemas, connection_schemas, connection_schemas, second_activations)

        with tf.name_scope('forward'):
            layers = convert(it2(it1(initial(inpt))))
            layers = it4(layers)
            layers = it5(layers)
            self.summaries.extend([image_summary("l00", layers[0].features),
                image_summary("l10", layers[1].features)
                                   ])
            layers = it4(layers)
            layers = it5(layers)
            self.summaries.extend([image_summary("l01", layers[0].features),
                image_summary("l11", layers[1].features)
                                   ])
            layers = it4(layers)
            layers = it5(layers)
            self.summaries.extend([image_summary("l02", layers[0].features),
                image_summary("l12", layers[1].features)
                                   ])
            pp.pprint(layers)
            vector_labels = layers[0].features
            print(vector_labels)

        var_list = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='params')
        reconstruct_var_list = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='reconstruct_params')

        with tf.name_scope("loss"):
            #loss1, prediction = self.label_loss_fun(
            #    vector_labels, human_labels)
            loss2, long_range_affinities = self.long_range_loss_fun(
                vector_labels, human_labels, offsets)
            #loss = tf.nn.sigmoid(
            #    (tf.to_float(self.step) - 75000) / 10000) * loss1 + loss2
            loss = loss2

        def training_iteration():
            optimizer = tf.train.AdamOptimizer(0.0001, epsilon=0.1)
            train_op = optimizer.minimize(loss)
            with tf.control_dependencies([train_op]):
                train_op = tf.group(self.step.assign_add(1), tf.Print(
                    0, [self.step, iteration_type, loss],
                    message="step|iteration_type|loss"))
                quick_summary_op = tf.merge_summary([
                    tf.scalar_summary("loss_train", loss),
                ])
            return train_op, quick_summary_op

        def test_iteration():
            quick_summary_op = tf.merge_summary(
                [tf.scalar_summary("loss_test", loss)])
            return tf.no_op(), quick_summary_op

        train_op, quick_summary_op = tf.cond(
            tf.equal(self.iteration_type, 0),
            training_iteration, test_iteration)

        self.summaries.extend(
            [image_slice_summary(
                "boundary_{}".format(key), long_range_affinities[key])
                for key in long_range_affinities])
        self.summaries.extend([image_summary("image", inpt),
                               # image_summary("vector_labels", vector_labels)
                               ])
        # self.summaries.extend([tf.image_summary("prediction",
        #                        tf.reshape(prediction,[1,maxn,maxn,1]))])
        summary_op = tf.merge_summary(self.summaries)

        init = tf.initialize_all_variables()
        self.sess.run(init)

        self.saver = tf.train.Saver()
        self.saver1 = tf.train.Saver(var_list=var_list)
        self.train_op = train_op
        self.quick_summary_op = quick_summary_op

        self.summary_op = summary_op
        self.inpt_placeholder = inpt_placeholder
        self.human_labels_placeholder = human_labels_placeholder
        self.iteration_type_placeholder = iteration_type_placeholder
        self.seed_placeholder = seed_placeholder

    def init_log(self):
        print ('How do you want to call this experiment?')
        date = datetime.now().strftime("%j-%H-%M-%S")
        exp_name = date+'-'+raw_input('run name: ')

        logdir = os.path.expanduser("~/experiments/{}/".format(exp_name))
        self.logdir = logdir

        print('logging to {}, also created a branch called {}'
              .format(logdir, exp_name))
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        save_experiment(exp_name)
        self.summary_writer = tf.train.SummaryWriter(
            logdir, graph=self.sess.graph)

    def restore(self, modelpath):
        modelpath = os.path.expanduser(modelpath)
        self.saver.restore(self.sess, modelpath)

    def restore1(self, modelpath):
        modelpath = os.path.expanduser(modelpath)
        self.saver1.restore(self.sess, modelpath)

    def restore2(self, modelpath):
        modelpath = os.path.expanduser(modelpath)
        self.saver2.restore(self.sess, modelpath)

    def affinity(self, x, y):
        """
        We want an array that is 1 where x and y values are the same
        and close to 0 where they are different.

        We do this for all the nvec_labels, meaning that interaction
        goes from [-nvec_labes, nvec_labels].

        The tensor return is a function of the interaction like this:
          1 ++-------+-------+--------+------****------+--------+-------+----++
            +        +       +        +    **  + **    + exp(-0.5*(x**2)) *** +
        0.9 ++                            **      **                         ++
            |                            *          *                         |
        0.8 ++                           *          *                        ++
            |                           *            *                        |
        0.7 ++                         *              *                      ++
            |                         *                *                      |
        0.6 ++                        *                *                     ++
        0.5 ++                      **                  **                   ++
            |                       *                    *                    |
        0.4 ++                     *                      *                  ++
            |                     *                        *                  |
        0.3 ++                    *                        *                 ++
            |                   **                          **                |
        0.2 ++                 **                            **              ++
            |                **                                **             |
        0.1 ++             ***                                  ***          ++
            +        +  **** +        +        +       +        + ****  +     +
          0 ************-----+--------+--------+-------+--------+-----*********
           -4       -3      -2       -1        0       1        2       3     4


        Args:
            x (tensor): 4 dimensional array of floats from 0 to 1
            y (tensor): 4 dimensional array of floats from 0 to 1

        Returns:
            tensor of floats: with ones where x == y and 0 where x ortogonal
            to y, and an smooth transition between both extreme conditions.
        """
        displacement = x - y
        interaction = tf.reduce_sum(
            displacement * displacement,
            reduction_indices=[3],
            keep_dims=True)
        return tf.exp(-0.5 * interaction)

    def selection_to_attention(self, vector_labels, selection):
        return tf.reduce_sum(selection * vector_labels,
                             reduction_indices=[0, 1, 2],
                             keep_dims=True) / tf.reduce_sum(selection)

    def attention_to_selection(self, vector_labels, attention):
        return self.affinity(vector_labels, attention)

    def long_range_loss_fun(self, vec_labels, human_labels, offsets):
        cost = 0
        otpts = {}

        if self.mask_boundary:
            mask = tf.to_float(tf.not_equal(self.human_labels, 0))
        else:
            mask = 1

        if self.mask_unknown:
            mask = mask * \
                tf.to_float(tf.not_equal(self.human_labels, self.maxn))
        else:
            mask = mask * 1

        for i, offset in enumerate(offsets):
            guess = self.affinity(
                *
                get_pair(
                    vec_labels,
                    offset,
                    self.patch_size))
            truth = self.label_diff(
                *
                get_pair(
                    human_labels,
                    offset,
                    self.patch_size))

            if mask != 1:
                mask1, mask2 = get_pair(mask, offset, self.patch_size)
            else:
                mask1, mask2 = 1, 1

            otpts[offset] = guess

            cost += tf.reduce_sum(mask1 * mask2 *
                                  bounded_cross_entropy(guess, truth))

        return cost, otpts

    def label_diff(self, x, y):
        return tf.to_float(tf.equal(x, y))

    def batch_interaction(self, mu0, cov0, mu1, cov1):
        nvec_labels = self.nvec_labels
        maxn = self.maxn
        A = tf.reshape(self.propagator, [1, 1, nvec_labels, nvec_labels])
        invA = tf.reshape(
            self.inv_propagator, [
                1, 1, nvec_labels, nvec_labels])
        mu0 = tf.reshape(mu0, [maxn, 1, nvec_labels, 1])
        mu1 = tf.reshape(mu1, [1, maxn, nvec_labels, 1])
        cov0 = tf.reshape(cov0, [maxn, 1, nvec_labels, nvec_labels])
        cov1 = tf.reshape(cov1, [1, maxn, nvec_labels, nvec_labels])
        identity = tf.reshape(tf.diag(tf.ones((nvec_labels,))), [
                              1, 1, nvec_labels, nvec_labels])
        identity2 = tf.reshape(identity_matrix(
            2 * nvec_labels), [1, 1, 2 * nvec_labels, 2 * nvec_labels])

        cov0 = cov0 + 0.00001 * identity
        cov1 = cov1 + 0.00001 * identity

        delta = mu1 - mu0
        delta2 = vec_cat(delta, delta)

        sqcov0 = tf.tile(tf.batch_cholesky(cov0), [1, maxn, 1, 1])
        sqcov1 = tf.tile(tf.batch_cholesky(cov1), [maxn, 1, 1, 1])
        invA = tf.tile(invA, [maxn, maxn, 1, 1])
        A = tf.tile(A, [maxn, maxn, 1, 1])

        scale = block_cat(
            sqcov0,
            tf.zeros_like(sqcov0),
            tf.zeros_like(sqcov0),
            sqcov1)
        M = batch_matmul(
            batch_transpose(scale),
            block_cat(
                invA,
                invA,
                invA,
                invA),
            scale) + identity2
        scale2 = block_cat(invA, tf.zeros_like(A), tf.zeros_like(A), invA)

        v = batch_matmul(
            tf.batch_matrix_inverse(M),
            batch_transpose(scale),
            scale2,
            delta2)

        ret1 = 1 / tf.sqrt(tf.batch_matrix_determinant(M))
        ret2 = tf.exp(-0.5 * (batch_matmul(batch_transpose(delta),
                      invA, delta) - batch_matmul(batch_transpose(v), M, v)))

        ret = ret1 * tf.squeeze(ret2, [2, 3])

        return ret

    def label_loss_fun(self, vec_labels, human_labels):
        maxn = self.maxn
        human_labels = tf.squeeze(human_labels, squeeze_dims=[3])

        populations = tf.dynamic_partition(
            vec_labels, human_labels, maxn + 1)[0:-1]
        weights = [tf.to_float(tf.shape(pop)[0]) for pop in populations]

        if self.mask_boundary:
            weights[0] = 0

        valid = [tf.greater(w, 1) for w in weights]
        means = conditional_map(
            lambda i: tf.reshape(
                tf.reduce_mean(populations[i], reduction_indices=[0]),
                [-1, 1]),
            xrange(maxn),
            valid,
            default=tf.zeros((self.nvec_labels, 1), dtype=tf.float32))

        covs = conditional_map(
            lambda i: covariance(
                populations[i], means[i], weights[i]),
            xrange(maxn), valid,
            default=tf.zeros(
                (self.nvec_labels, self.nvec_labels), dtype=tf.float32))

        pack_weights = tf.pack(weights)
        pack_means = tf.pack(means)
        pack_covs = tf.pack(covs)

        predictions = self.batch_interaction(
            pack_means, pack_covs, pack_means, pack_covs)
        predictions = 0.99998 * predictions + 0.00001
        objective = identity_matrix(maxn)
        weight_matrix = tf.reshape(
            tf.sqrt(pack_weights), [-1, 1]) * tf.reshape(
            tf.sqrt(pack_weights), [1, -1])

        cost = - objective * tf.log(predictions) - \
            (1 - objective) * tf.log(1 - predictions)

        return tf.reduce_sum(weight_matrix * cost), predictions

    def train(
            self,
            train_dataset,
            test_dataset,
            nsteps=100000,
            checkpoint_interval=250):
        self.init_log()
        print ('log initialized')
        t = threading.Thread(
            target=Model.enqueuer, args=(
                self, train_dataset, test_dataset))
        t.daemon = True
        t.start()
        for i in xrange(nsteps):
            t = time.time()
            _, quick_summary = self.sess.run(
                [self.train_op, self.quick_summary_op])
            elapsed = time.time() - t
            print("elapsed: ", elapsed)
            self.summary_writer.add_summary(
                quick_summary, self.sess.run(self.step))
            if i % checkpoint_interval == 0:
                print("checkpointing")
                step = self.sess.run(self.step)
                print("logging")
                self.saver.save(
                    self.sess,
                    self.logdir +
                    "model" +
                    str(step) +
                    ".ckpt")
                self.summary_writer.add_summary(
                    self.sess.run(self.summary_op), step)
                self.summary_writer.flush()
                print("done")

    def enqueuer(self, train_dataset, test_dataset):
        train_generator = gen(train_dataset, self.patch_size, self.maxn)
        test_generator = gen(test_dataset, self.patch_size, self.maxn)
        for label, (image, human_labels) in alternating_iterator(
                [train_generator, test_generator], [20, 1], label=True):
            feed_dict = {
                self.inpt_placeholder: pad_shape(image),
                self.human_labels_placeholder: pad_shape(human_labels),
                self.iteration_type_placeholder: label}
            self.sess.run(self.enqueue_op, feed_dict=feed_dict)

    def compute_vector_labels(self, inpt):
        dummy_human_labels = np.zeros(self.patch_size, dtype=np.int32)
        feed_dict = {
            self.inpt_placeholder: pad_shape(inpt),
            self.human_labels_placeholder: pad_shape(dummy_human_labels),
            self.iteration_type_placeholder: 1}
        self.sess.run(self.enqueue_op, feed_dict=feed_dict)
        return self.sess.run(self.vector_labels)


def length_scale(x):
    if x == 0:
        return -1
    else:
        return math.log(abs(x))


def valid_pair(x, y, strict=False):
    return x == 0 or y == 0 or (
        (not strict or length_scale(x) >= length_scale(y)) and abs(
            length_scale(x) - length_scale(y)) <= math.log(3.1))


def valid_offset(x):
    return x > (
        0,
        0,
        0) and valid_pair(
        4 *
        x[0],
        x[1],
        strict=True) and valid_pair(
            4 *
            x[0],
            x[2],
            strict=True) and valid_pair(
                x[1],
        x[2])

args = {
    "offsets": filter(valid_offset, itertools.product(
        [-3, -1, 0, 1, 3],
        [-27, -9, -3, -1, 0, 1, 3, 9, 27],
        [-27, -9, -3, -1, 0, 1, 3, 9, 27])),
    "devices": ["/gpu:0"],
    "patch_size": (32, 158, 158),
    "mask_boundary": False,
    "mask_unknown": True,
    "nvec_labels": 6,
    "maxn": 40,
}
# offsets:
# devices: device used by tensorflow to run the model
# patch_size: size of window input and output by the model
# mask_boundary:
# mask_unkown
# nvec_labels: size of the vectors of every pixel
# nrecounstrctions

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(args)
with tf.device(args["devices"][0]):
    main_model = Model(**args)

if __name__ == '__main__':
    TRAIN = Dataset(SNEMI3D_TRAIN_DIR, image=True, human_labels=True)
    TEST = Dataset(SNEMI3D_TEST_DIR, image=True, human_labels=True)

    #main_model.restore1(os.path.expanduser("/usr/people/jzung/experiments/270-17-22-08-/model82621.ckpt"))

    main_model.train(train_dataset=TRAIN, test_dataset=TEST, nsteps=1000000)
    trace = timeline.Timeline(step_stats=main_model.run_metadata.step_stats)
    trace_file = open('timeline.ctf.json', 'w')
    trace_file.write(trace.generate_chrome_trace_format())
    print("done")

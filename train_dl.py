# -*- coding: utf-8 -*-
import os
import sys
import csv
import time
import json
import datetime
import pickle as pkl
import tensorflow as tf
from tensorflow.contrib import learn

from DataLoader import DataLoader
from clstm_classifier import clstm_clf
from cnn_classifier import cnn_clf
from configuration import clstm_config, cnn_config
from sklearn.metrics import precision_recall_fscore_support


model = 'clstm'
if model == 'clstm':
    config = clstm_config()
elif model == 'cnn':
    config = cnn_config()
data_loader = DataLoader(config)
# Output files directory
timestamp = str(int(time.time()))
outdir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
if not os.path.exists(outdir):
    os.makedirs(outdir)


# Train
# =============================================================================

with tf.Graph().as_default():
    with tf.Session() as sess:
        if model == 'clstm':
            classifier = clstm_clf(config)
        elif model == 'cnn':
            classifier = cnn_clf(config)

        # Train procedure
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Learning rate decay
        starter_learning_rate = config.learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                   global_step,
                                                   config.decay_steps,
                                                   config.decay_rate,
                                                   staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(classifier.cost)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Summaries
        loss_summary = tf.summary.scalar('Loss', classifier.cost)
        accuracy_summary = tf.summary.scalar('Accuracy', classifier.accuracy)

        # Train summary
        train_summary_op = tf.summary.merge_all()
        train_summary_dir = os.path.join(outdir, 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Test summary
        test_summary_op = tf.summary.merge_all()
        test_summary_dir = os.path.join(outdir, 'summaries', 'test')
        test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

        saver = tf.train.Saver(max_to_keep = config.num_checkpoint)

        sess.run(tf.global_variables_initializer())


        def run_step(model, input_x, input_y, is_training=True):
            """Run one step of the training process."""
            #input_x, input_y, sequence_length = input_data

            fetches = {'step': global_step,
                       'cost': classifier.cost,
                       'accuracy': classifier.accuracy,
                       'predictions': classifier.predictions,
                       'learning_rate': learning_rate}
            feed_dict = {classifier.input_x: input_x,
                         classifier.input_y: input_y}

            if model == 'clstm':
                fetches['final_state'] = classifier.final_state
                feed_dict[classifier.batch_size] = len(input_x)
            elif model == 'cnn':
                pass

            if is_training:
                fetches['train_op'] = train_op
                fetches['summaries'] = train_summary_op
                feed_dict[classifier.keep_prob] = config.keep_prob
            else:
                fetches['summaries'] = test_summary_op
                feed_dict[classifier.keep_prob] = 1.0

            vars = sess.run(fetches, feed_dict)
            step = vars['step']
            cost = vars['cost']
            accuracy = vars['accuracy']
            predictions = vars['predictions']
            summaries = vars['summaries']

            precision, recall, f1, _ = precision_recall_fscore_support(input_y, predictions, average = 'binary')
            # Write summaries to file
            if is_training:
                train_summary_writer.add_summary(summaries, step)
            else:
                test_summary_writer.add_summary(summaries, step)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step: {}, loss: {:g}, accuracy: {:g}, precision: {:g}, recall: {:g}, f1: {:g}".format(time_str, step, cost, accuracy, precision, recall, f1))

            return accuracy


        print('Start training ...')

        for i in range(config.num_epochs):
            for j in range(data_loader.num_batches):
                input_x, input_y = data_loader.next_batch()
                run_step(model, input_x, input_y, is_training = True)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % config.evaluate_every_steps == 0:
                    print('\nTest')
                    input_x, input_y = data_loader.get_test_data()
                    run_step(model, input_x, input_y, is_training=False)
                    print('')
                if current_step % config.save_every_steps == 0:
                    save_path = saver.save(sess, os.path.join(outdir, 'model/clf'), current_step)

        '''
        for train_input in train_data:
            run_step(train_input, is_training=True)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % FLAGS.evaluate_every_steps == 0:
                print('\nValidation')
                run_step((x_valid, y_valid, valid_lengths), is_training=False)
                print('')

            if current_step % FLAGS.save_every_steps == 0:
                save_path = saver.save(sess, os.path.join(outdir, 'model/clf'), current_step)
        '''

        print('\nAll the files have been saved to {}\n'.format(outdir))

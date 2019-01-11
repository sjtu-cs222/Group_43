import time
import numpy as np
import tensorflow as tf
import logging
import importlib
import sys
import gan.kdd_utilities as network
import data.kdd as data
from sklearn.metrics import precision_recall_fscore_support
from configuration import generator_config
from configuration import discriminator_config
from configuration import gan_training_config
from configuration import clstm_config
from generator import Generator
from discriminator import Discriminator

RANDOM_SEED = 146
FREQ_PRINT = 20 # print frequency image tensorboard [20]
STEPS_NUMBER = 500

def get_getter(ema):  # to update neural net with moving avg variables, suitable for ss learning cf Saliman
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var

    return ema_getter

def display_parameters(batch_size, starting_lr, ema_decay, weight, method, degree):
    '''See parameters
    '''
    print('Batch size: ', batch_size)
    print('Starting learning rate: ', starting_lr)
    print('EMA Decay: ', ema_decay)
    print('Weight: ', weight)
    print('Method for discriminator: ', method)
    print('Degree for L norms: ', degree)

def display_progression_epoch(j, id_max):
    '''See epoch progression
    '''
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush

def create_logdir(method, weight, rd):
    """ Directory to save training logs, weights, biases, etc."""
    return "gan/train_logs/kdd/{}/{}/{}".format(weight, method, rd)

gen_config = generator_config()
dis_config = discriminator_config()
train_config = gan_training_config()
data_config = clstm_config()
data_loader = DataLoader(data_config)
rng = np.random.RandomState(RANDOM_SEED)

with tf.Graph().as_default():
    with tf.Session() as sess:
        generator = Generator(gen_config)
        discriminator = Discriminator(dis_config)
        generator.build()

        #Pretrain csltm classifier in generator
        # Train procedure
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Learning rate decay
        starter_learning_rate = train_config.pretrain_learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                   global_step,
                                                   train_config.decay_steps,
                                                   train_config.decay_rate,
                                                   staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(generator.pretrain_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        sess.run(tf.global_variables_initializer())

        def pretrain_run_step(model, input_x, input_y, is_training=True):
            """Run one step of the training process."""
            #input_x, input_y, sequence_length = input_data

            fetches = {'step': global_step,
                       'cost': generator.pretrain_loss,
                       'accuracy': generator.pretrain_accuracy,
                       'predictions': generator.pretrain_predictions,
                       'learning_rate': learning_rate}
            feed_dict = {generator.pretrain_input_x: input_x,
                         generator.pretrain_input_y: input_y}

            if model == 'clstm':
                fetches['final_state'] = classifier.final_state
                feed_dict[generator.pretrain_batch_size] = len(input_x)
            elif model == 'cnn':
                pass
            if is_training:
                fetches['train_op'] = train_op
                feed_dict[generator.pretrain_keep_prob] = generator_.keep_prob
            else:
                feed_dict[generator.pretrain_keep_prob] = 1.0

            vars = sess.run(fetches, feed_dict)
            step = vars['step']
            cost = vars['cost']
            accuracy = vars['accuracy']
            predictions = vars['predictions']

            precision, recall, f1, _ = precision_recall_fscore_support(input_y, predictions, average = 'binary')
            # Write summaries to file

            time_str = datetime.datetime.now().isoformat()
            print("{}: step: {}, loss: {:g}, accuracy: {:g}, precision: {:g}, recall: {:g}, f1: {:g}".format(time_str, step, cost, accuracy, precision, recall, f1))

            return accuracy


        print('Start pretraining ...')

        for i in range(train_config.pretrain_num_epochs):
            for j in range(data_loader.num_batches):
                input_x, input_y = data_loader.next_batch()
                run_step('clstm', input_x, input_y, is_training = True)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % train_config.evaluate_every_steps == 0:
                    print('\nTest')
                    input_x, input_y = data_loader.get_test_data()
                    run_step('clstm', input_x, input_y, is_training=False)
                    print('')
                    

#Adversarial training
data_loader.reset_batch()
real_data = tf.placeholder(tf.float32, shape=[None, generator_config.data_window, generator_config.feature_size, name="real_input")
is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
learning_rate = tf.placeholder(tf.float32, shape=(), name="lr_pl")

starting_lr = train_config.adversarial_learning_rate
latent_dim = train_config.latent_dim
batch_size = train_config.batch_size
ema_decay = 0.9999

random_z = tf.random_normal([batch_size*generator_config.data_window, latent_dim], mean=0.0, stddev=1.0, name='random_z')
generated_data = generator.generate_fake_sample(random_z, is_training=is_training_pl, batch_size) #[batch_size, data_window, feature_size]
real_lstm_outputs, real_loss = generator.classifier_network(real_data)
fake_lstm_outputs, fake_loss = generator.classifier_network(generated_data) #[batch_size, sequence_length, hidden_size]
real_d, inter_layer_real = discriminator.discriminator_network(real_lstm_outputs, is_training=is_training_pl)
fake_d, inter_layer_fake = discriminator.discriminator_network(fake_lstm_outputs, is_training=is_training_pl, reuse=True)

with tf.name_scope('loss_functions'):
    # Calculate seperate losses for discriminator with real and fake images
    real_discriminator_loss = tf.losses.sigmoid_cross_entropy(tf.constant(1, shape=[batch_size]), real_d, scope='real_discriminator_loss')
    fake_discriminator_loss = tf.losses.sigmoid_cross_entropy(tf.constant(0, shape=[batch_size]), fake_d, scope='fake_discriminator_loss')
    # Add discriminator losses
    discriminator_loss = real_discriminator_loss + fake_discriminator_loss
    # Calculate loss for generator by flipping label on discriminator output
    generator_loss = tf.losses.sigmoid_cross_entropy(tf.constant(1, shape=[batch_size]), fake_d, scope='generator_loss') * fake_loss

def get_variable_via_scope(scope_lst):
    vars = []
    for sc in scope_lst:
        sc_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)
        vars.extend(sc_variable)
    return vars

with tf.name_scope('optimizers'):
# control op dependencies for batch norm and trainable variables
dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
gvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

update_ops_gen = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
update_ops_dis = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')

optimizer_dis = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='dis_optimizer')
optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='gen_optimizer')

no_change_scope = ['teller']
no_change_vars = get_variable_via_scope(no_change_scope)
for v in no_change_vars:
    gvars.remove(v)
with tf.control_dependencies(update_ops_gen): # attached op for moving average batch norm
    gen_op = optimizer_gen.minimize(generator_loss, var_list=gvars)
with tf.control_dependencies(update_ops_dis):
    dis_op = optimizer_dis.minimize(discriminator_loss, var_list=dvars)

dis_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
maintain_averages_op_dis = dis_ema.apply(dvars)

with tf.control_dependencies([dis_op]):
    train_dis_op = tf.group(maintain_averages_op_dis)

gen_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
maintain_averages_op_gen = gen_ema.apply(gvars)

with tf.control_dependencies([gen_op]):
    train_gen_op = tf.group(maintain_averages_op_gen)

with tf.name_scope('training_summary'):
    with tf.name_scope('dis_summary'):
        tf.summary.scalar('real_discriminator_loss', real_discriminator_loss, ['dis'])
        tf.summary.scalar('fake_discriminator_loss', fake_discriminator_loss, ['dis'])
        tf.summary.scalar('discriminator_loss', discriminator_loss, ['dis'])

    with tf.name_scope('gen_summary'):
        tf.summary.scalar('loss_generator', generator_loss, ['gen'])


    sum_op_dis = tf.summary.merge_all('dis')
    sum_op_gen = tf.summary.merge_all('gen')


with tf.variable_scope("latent_variable"):
    z_optim = tf.get_variable(name='z_optim', shape= [batch_size, latent_dim], initializer=tf.truncated_normal_initializer())
    reinit_z = z_optim.initializer
# EMA
generator_ema = generator.generate_fake_sample(z_optim, is_training=is_training_pl, getter=get_getter(gen_ema), reuse=True, batch_size)
# Pass real and fake images into discriminator separately
real_d_ema, inter_layer_real_ema = discriminator.discriminator_network(input_pl, is_training=is_training_pl, getter=get_getter(gen_ema), reuse=True)
fake_d_ema, inter_layer_fake_ema = discriminator.discriminator_network(generator_ema, is_training=is_training_pl, getter=get_getter(gen_ema), reuse=True)

with tf.name_scope('error_loss'):
    delta = real_data - generator_ema
    delta_flat = tf.contrib.layers.flatten(delta)
    gen_score = tf.norm(delta_flat, ord=degree, axis=1, keep_dims=False, name='epsilon')

with tf.variable_scope('Discriminator_loss'):
    if method == "cross-e":
        dis_score = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(fake_d_ema), logits=fake_d_ema)

    elif method == "fm":
        fm = inter_layer_real_ema - inter_layer_fake_ema
        fm = tf.contrib.layers.flatten(fm)
        dis_score = tf.norm(fm, ord=degree, axis=1, keep_dims=False,
                             name='d_loss')

    dis_score = tf.squeeze(dis_score)

with tf.variable_scope('Total_loss'):
    loss = (1 - weight) * gen_score + weight * dis_score

with tf.variable_scope("Test_learning_rate"):
    step = tf.Variable(0, trainable=False)
    boundaries = [300, 400]
    values = [0.01, 0.001, 0.0005]
    learning_rate_invert = tf.train.piecewise_constant(step, boundaries, values)
    reinit_lr = tf.variables_initializer(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope="Test_learning_rate"))

with tf.name_scope('Test_optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate_invert).minimize(loss, global_step=step, var_list=[z_optim], name='optimizer')
    reinit_optim = tf.variables_initializer(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope='Test_optimizer'))

reinit_test_graph_op = [reinit_z, reinit_lr, reinit_optim]

with tf.name_scope("Scores"):
    list_scores = loss

logdir = create_logdir(method, weight, random_seed)

sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=None,
                             save_model_secs=120)

logger.info('Start training...')
with sv.managed_session() as sess:

    logger.info('Initialization done')

    writer = tf.summary.FileWriter(logdir, sess.graph)

    train_batch = 0
    epoch = 0

    while not sv.should_stop() and epoch < nb_epochs:

        lr = starting_lr

        begin = time.time()
        trainx = trainx[rng.permutation(trainx.shape[0])]  # shuffling unl dataset
        trainx_copy = trainx_copy[rng.permutation(trainx.shape[0])]

        train_loss_dis, train_loss_gen = [0, 0]
        # training
        for t in range(nr_batches_train):
            display_progression_epoch(t, nr_batches_train)

            # construct randomly permuted minibatches
            ran_from = t * batch_size
            ran_to = (t + 1) * batch_size

            # train discriminator
            feed_dict = {input_pl: trainx[ran_from:ran_to],
                            is_training_pl:True,
                            learning_rate:lr}
            _, ld, sm = sess.run([train_dis_op, discriminator_loss, sum_op_dis], feed_dict=feed_dict)
            train_loss_dis += ld
            writer.add_summary(sm, train_batch)

            # train generator
            feed_dict = {input_pl: trainx_copy[ran_from:ran_to],
                             is_training_pl:True,
                             learning_rate:lr}
            _, lg, sm = sess.run([train_gen_op, generator_loss, sum_op_gen], feed_dict=feed_dict)
            train_loss_gen += lg
            writer.add_summary(sm, train_batch)

            train_batch += 1

        train_loss_gen /= nr_batches_train
        train_loss_dis /= nr_batches_train

        logger.info('Epoch terminated')
        print("Epoch %d | time = %ds | loss gen = %.4f | loss dis = %.4f "
                  % (epoch, time.time() - begin, train_loss_gen, train_loss_dis))

        epoch += 1

    logger.warn('Testing evaluation...')
    inds = rng.permutation(testx.shape[0])
    testx = testx[inds]  # shuffling unl dataset
    testy = testy[inds]
    scores = []
    inference_time = []

    # Testing
    for t in range(nr_batches_test):

        # construct randomly permuted minibatches
        ran_from = t * batch_size
        ran_to = (t + 1) * batch_size
        begin_val_batch = time.time()

        # invert the gan
        feed_dict = {input_pl: testx[ran_from:ran_to],
                         is_training_pl:False}

        for step in range(STEPS_NUMBER):
            _ = sess.run(optimizer, feed_dict=feed_dict)
        scores += sess.run(list_scores, feed_dict=feed_dict).tolist()
        inference_time.append(time.time() - begin_val_batch)
        sess.run(reinit_test_graph_op)

    logger.info('Testing : mean inference time is %.4f' % (
        np.mean(inference_time)))
    ran_from = nr_batches_test * batch_size
    ran_to = (nr_batches_test + 1) * batch_size
    size = testx[ran_from:ran_to].shape[0]
    fill = np.ones([batch_size - size, 121])

    batch = np.concatenate([testx[ran_from:ran_to], fill], axis=0)
    feed_dict = {input_pl: batch,
                     is_training_pl: False}

    for step in range(STEPS_NUMBER):
        _ = sess.run(optimizer, feed_dict=feed_dict)
    batch_score = sess.run(list_scores,
                           feed_dict=feed_dict).tolist()

    scores += batch_score[:size]

    per = np.percentile(scores, 80)

    y_pred = scores.copy()
    y_pred = np.array(y_pred)

    inds = (y_pred < per)
    inds_comp = (y_pred >= per)

    y_pred[inds] = 0
    y_pred[inds_comp] = 1

    precision, recall, f1,_ = precision_recall_fscore_support(testy,
                                                                  y_pred,
                                                                  average='binary')
    print("Testing : Prec = %.4f | Rec = %.4f | F1 = %.4f "
            % (precision, recall, f1))



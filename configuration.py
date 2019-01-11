class clstm_config(object):
    """Wrapper calss for C-LSTM hyperparameter"""

    def __init__(self):
        #data hyperparameter
        self.data_window = 60 #dimention of data window
        self.feature_size = 1 #dimention of feature of each data
        self.padding = 1
        self.is_shuffle = True
        self.data_file_path = "A1Benchmark"

        #model hyperparameter
        self.model = 'clstm'
        self.num_classes = 2 #dimention of class
        self.filter_size = 5 #CNN filter window size
        self.num_filters = 32 #CNN filter number
        self.hidden_size = 64 #dimension of hidden unit in LSTM
        self.num_layers = 2 #number for layers in LSTM, 2 means biLSTM
        self.dense_size = 32 #dimension of fully connected layer
        self.l2_reg_lambda = 0.0d01 #L2 regularization strength
        self.keep_prob = 0.5 #Dropout keep probability'

        #training hyperparameter
        self.learning_rate = 1e-3 #learning rate
        self.decay_steps = 100000 #Learning rate decay steps
        self.decay_rate = 1 #Learning rate decay rate. Range: (0, 1]
        self.batch_size = 512 #Batch size
        self.num_epochs = 50 #Number of epochs
        self.save_every_steps = 1000 #Save the model after this many steps
        self.num_checkpoint = 10 #Number of models to store'
        self.evaluate_every_steps = 100 #Evaluate the model on validation set after this many steps

class cnn_config(object):
    
    def __init__(self):
        #data hyperparameter
        self.data_window = 60 #dimention of data window
        self.feature_size = 1 #dimention of feature of each data
        self.padding = 1
        self.is_shuffle = True
        self.data_file_path = "A1Benchmark"

        #model hyperparameter
        self.model = 'cnn'
        self.num_classes = 2 #dimention of class
        self.filter_size = 5 #CNN filter window size
        self.num_filters = 64 #CNN filter number
        self.l2_reg_lambda = 0.001 #L2 regularization strength
        self.keep_prob = 0.5 #Dropout keep probability'

        #training hyperparameter
        self.learning_rate = 1e-3 #learning rate
        self.decay_steps = 100000 #Learning rate decay steps
        self.decay_rate = 1 #Learning rate decay rate. Range: (0, 1]
        self.batch_size = 512 #Batch size
        self.num_epochs = 50 #Number of epochs
        self.save_every_steps = 1000 #Save the model after this many steps
        self.num_checkpoint = 10 #Number of models to store'
        self.evaluate_every_steps = 100 #Evaluate the model on validation set after this many steps


class generator_config(object):
    """Wrapper class for generator hyperparameter"""

    def __init__(self):
        #data hyperparameter
        self.data_window = 60 #dimention of data window
        self.feature_size = 1 #dimention of feature of each data

        #model hyperparameter
        self.num_classes = 2 #dimention of class
        self.filter_size = 5 #CNN filter window size
        self.num_filters = 32 #CNN filter number
        self.hidden_size = 64 #dimension of hidden unit in LSTM
        self.num_layers = 2 #number for layers in LSTM, 2 means biLSTM
        self.dense_size = 32 #dimension of fully connected layer
        self.l2_reg_lambda = 0.001 #L2 regularization strength
        self.keep_prob = 0.5 #Dropout keep probability'
        
        #generation hyperparameter
        latent_dim = 32
        self.init_kernel = tf.contrib.layers.xavier_initializer()


class discriminator_config(object):
    """Wrapper class for discriminator hyperparameter"""

    def __init__(self):
        self.dis_inter_layer_dim = 128
        self.init_kernel = tf.contrib.layers.xavier_initializer()


class gan_training_config(object):
    """Wrapper class for parameters for training"""

    def __init__(self):
        self.batch_size = 512 #Batch size
        self.pretrain_learning_rate = 0.001
        self.learning_rate = 0.00001
        self.decay_steps = 100000 #Learning rate decay steps
        self.decay_rate = 1 #Learning rate decay rate. Range: (0, 1]
        self.pretrain_num_epochs = 50
        self.pretrain_num_checkpoint = 10 #Number of models to store'
        self.pretrain_evaluate_every_steps = 100 #Evaluate the model on validation set after this many steps
        
        self.adversarial_learning_rate = 0.00001
        self.latent_dim = 32
        self.gen_learning_rate = 0.001 #learning rate of generator X
        self.gen_update_time = 1 #update times of generator in adversarial training
        self.dis_update_time_adv = 5 #update times of discriminator in adversarial training
        self.dis_update_epoch_adv = 3 #update epoch / times of discriminator
        self.dis_update_time_pre = 50 #pretraining times of discriminator
        self.dis_update_epoch_pre = 3 #number of epoch / time in pretraining
        self.pretrained_epoch_num = 120 #Number of pretraining epoch
        self.rollout_num = 16  #Rollout number for reward estimation
        self.test_per_epoch = 5 #Test the NLL per epoch
        self.batch_size = 64 #Batch size used for training
        self.save_pretrained = 120 # Whether to save model in certain epoch (optional)
        self.grad_clip = 5.0 #Gradient Clipping X
        self.seed = 88 #Random seed used for initialization
        self.start_token = 0 #special start token
        self.total_batch = 200 #total batch used for adversarial training
        self.positive_file = "save/real_data.txt"  # save path of real data generated by target LSTM
        self.negative_file = "save/generator_sample.txt" #save path of fake data generated by generator
        self.eval_file = "save/eval_file.txt" #file used for evaluation
        self.generated_num = 10000 #Number of samples from generator used for evaluation

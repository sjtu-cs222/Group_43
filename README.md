## Anomaly Detection with seqGAN + RL

#### Model 

Overall model: GAN + RL

Generator: CNN + LSTM

Discriminator: DNN

![Blank Diagram](https://github.com/sjtu-cs222/Group_43/blob/master/Report/gan.png)

#### Data

We use the Yahoo Webscope S5 dataset, which is provided as part of the Yahoo! Webscope program. It consists of four classes and we only utilize the A1 class. A1 class is a sequence of data with only one feature arranged in time series. Sliding window algorithm is used for extracting data sequences with time features. Implementation details can be seen in DataLoader.py

#### Experiment

We did experiements on LSTM+DNN, CNN+DNN, CNN+LSTM+DNN, CNN+LSTM+GAN+RL. Achieving different performances. Details can be seen in our report.

#### How  to use

To train deep learning model, please run train_dl.py, to train GAN+RL model, please run train_gan.py








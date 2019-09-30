import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed'''
import time

import tensorflow as tf

from config import argparser
from data import get_dataloader
from model import Base, DIN, DIEN
from utils import eval

# Config
print(tf.__version__)
print("GPU Available: ", tf.test.is_gpu_available())

args = argparser()

# Data Load
train_data, test_data, \
user_count, item_count, cate_count, \
cate_list = get_dataloader(args.train_batch_size, args.test_batch_size)

# Loss, Optim
optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=0.0)
loss_metric = tf.keras.metrics.Sum()
auc_metric = tf.keras.metrics.AUC()

# Model
model = Base(user_count, item_count, cate_count, cate_list,
             args.user_dim, args.item_dim, args.cate_dim, args.dim_layers)

# Board
train_summary_writer = tf.summary.create_file_writer(args.log_path)

#@tf.function
def train_one_step(u,i,y,hist_i,sl):
    with tf.GradientTape() as tape:
        output,_ = model(u,i,hist_i,sl)
        loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=output,
                                                        labels=tf.cast(y, dtype=tf.float32)))
    gradient = tape.gradient(loss, model.trainable_variables)
    clip_gradient, _ = tf.clip_by_global_norm(gradient, 5.0)
    optimizer.apply_gradients(zip(clip_gradient, model.trainable_variables))

    loss_metric(loss)

# Train
def train(optimizer):
    best_loss= 0.
    best_auc = 0.
    start_time = time.time()
    for epoch in range(args.epochs):
        for step, (u, i, y, hist_i, sl) in enumerate(train_data, start=1):
            train_one_step(u, i, y, hist_i, sl)

            if step % args.print_step == 0:
                test_gauc, auc = eval(model, test_data)
                print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_GAUC: %.4f\tEval_AUC: %.4f' %
                      (epoch, step, loss_metric.result() / args.print_step, test_gauc, auc))

                if best_auc < test_gauc:
                    best_loss= loss_metric.result() / args.print_step
                    best_auc = test_gauc
                    model.save_weights(args.model_path+'cp-%d.ckpt'%epoch)
                loss_metric.reset_states()

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', best_loss, step=epoch)
            tf.summary.scalar('test_gauc', best_auc, step=epoch)

        loss_metric.reset_states()
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0)

        print('Epoch %d DONE\tCost time: %.2f' % (epoch, time.time()-start_time))
    print('Best test_gauc: ', best_auc)


# Main
if __name__ == '__main__':
    train(optimizer)

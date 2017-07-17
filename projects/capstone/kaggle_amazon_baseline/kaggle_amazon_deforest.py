import os
import os.path
import math
from datetime import datetime

import numpy as np
import tensorflow as tf
import sklearn.metrics as metrics

import process_input

#%%
BATCH_SIZE = 128
#12650
#15812
tf.app.flags.DEFINE_integer('how_many_training_steps', 15812,
                            """How many training steps to run before ending.""")
tf.app.flags.DEFINE_float('learning_rate', 0.0001,
                          """How large a learning rate to use when training.""")
tf.app.flags.DEFINE_integer('eval_step_interval', 100,
                            """How often to evaluate the training results.""")

tf.app.flags.DEFINE_string('log_dir', './log', """Path to log directory.""")
tf.app.flags.DEFINE_string('training_data', 'train_full_data.csv', """input training data""")

tf.app.flags.DEFINE_boolean('is_eval', False, """run evaluation""")
tf.app.flags.DEFINE_boolean('is_test', False, """run test""")

def inference(images, is_training):
    '''
    Args:
        images: 4D tensor [batch_size, img_width, img_height, img_channel]
    Notes:
        In each conv layer, the kernel size is:
        [kernel_size, kernel_size, number of input channels, number of output channels].
        number of input channels are from previuous layer, if previous layer is THE input
        layer, number of input channels should be image's channels.
        
            
    '''
    #conv1, [5, 5, 3, 96], The first two dimensions are the patch size,
    #the next is the number of input channels, 
    #the last is the number of output channels
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights', 
                                  shape = [3, 3, 3, 96],
                                  dtype = tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.05,dtype=tf.float32)) 
        biases = tf.get_variable('biases', 
                                 shape=[96],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name= scope.name)
    
    #pool1 and norm1   
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1],strides=[1,2,2,1],
                               padding='SAME', name='pooling1')
        if is_training:
            pool1 = tf.nn.dropout(pool1, keep_prob=1.)
    
    #conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,96, 64],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.05,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[64], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')
    
    #pool2
    with tf.variable_scope('pooling2_lrn') as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1], strides=[1,1,1,1],
                               padding='SAME',name='pooling2')
        if is_training:
            pool2 = tf.nn.dropout(pool2, keep_prob=1.)
    
    #local3
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[BATCH_SIZE, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim,384],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.05,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[384],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        if is_training:
            local3 = tf.nn.dropout(local3, keep_prob=0.7)
    
    
    #local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[384,192],
                                  dtype=tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.05,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[192],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')
        if is_training:
            local4 = tf.nn.dropout(local4, keep_prob=0.7)
        
    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[192, process_input.NUM_CLASSES],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.05,dtype=tf.float32))
        biases = tf.get_variable('biases', 
                                 shape=[process_input.NUM_CLASSES],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        logits = tf.add(tf.matmul(local4, weights), biases, name='logits')

    return logits

def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        labels = tf.cast(labels, tf.float32)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                name='sigmoid_cross_entropy_with_logits')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.summary.scalar('cross_entropy_mean', cross_entropy_mean)
    return cross_entropy_mean

def get_final_tensor(logits):
    final_tensor = tf.nn.sigmoid(logits, name='final_tensor')
    return final_tensor

def add_evaluation_step(result_tensor, labels):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.round(result_tensor), labels)
        with tf.name_scope('accuracy'):
            # Mean accuracy over all labels:
            # http://stackoverflow.com/questions/37746670/tensorflow-multi-label-accuracy-calculation
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)
    return evaluation_step

def round_prediction(result_tensor):
    with tf.name_scope('prediction'):
        correct_prediction = tf.round(result_tensor)
    return correct_prediction

def train():
    
    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    
    images, labels, images_names = process_input.inputs_pipeline(tf.app.flags.FLAGS.training_data, BATCH_SIZE, None)
    
    logits = inference(images, is_training=True)
    loss = losses(logits, labels) 
    
    #optimizer = tf.train.GradientDescentOptimizer(tf.app.flags.FLAGS.learning_rate)
    optimizer = tf.train.AdamOptimizer(tf.app.flags.FLAGS.learning_rate)
    train_op = optimizer.minimize(loss, global_step=my_global_step)

    final_tensor = get_final_tensor(logits)
    evaluation_op = add_evaluation_step(final_tensor, labels)
    round_prediction_op = round_prediction(final_tensor)
    
    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()
    
    sess = tf.Session()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    summary_writer = tf.summary.FileWriter(tf.app.flags.FLAGS.log_dir, sess.graph)
    
    try:
        for step in np.arange(tf.app.flags.FLAGS.how_many_training_steps):
            if coord.should_stop():
                    break
            train_summary, _ = sess.run([summary_op, train_op])
            summary_writer.add_summary(train_summary, step)
           
            # Every so often, print out how well the graph is training.
            is_last_step = (step + 1 == tf.app.flags.FLAGS.how_many_training_steps)
            if (step % tf.app.flags.FLAGS.eval_step_interval) == 0 or is_last_step:
                train_accuracy, cross_entropy_value = sess.run([evaluation_op, loss])
                print('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), step,
                                                                train_accuracy * 100))
                print('%s: Step %d: Cross entropy = %f' % (datetime.now(), step, cross_entropy_value))

            if step % 2000 == 0 or (step + 1) == tf.app.flags.FLAGS.how_many_training_steps:
                checkpoint_path = os.path.join(tf.app.flags.FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()

def evaluate():
    with tf.Graph().as_default():
        n_test = 8097
        images, labels, image_names = process_input.inputs_pipeline('test_data.csv', BATCH_SIZE, None)
        
        logits = inference(images, is_training=False)
        final_tensor = get_final_tensor(logits)
        evaluation_op = add_evaluation_step(final_tensor, labels)
        pred_op = round_prediction(final_tensor)

        saver = tf.train.Saver(tf.global_variables())
        
        with tf.Session() as sess:
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(tf.app.flags.FLAGS.log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return
        
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)
            
            try:
                num_iter = int(math.ceil(n_test/ BATCH_SIZE))
                total_sample_count = num_iter * BATCH_SIZE
                step = 0

                f2_scores = []
                while step < num_iter and not coord.should_stop():
                    _, prediction, accuracy, y, y_pred, name = sess.run([logits, final_tensor, evaluation_op, labels, pred_op, image_names])

                    _labels = y.astype(np.int)
                    _predictions = y_pred.astype(np.int)
                    #_accuracy = np.mean(np.equal(_labels, _predictions).astype(np.float))
                    _f2_score = metrics.fbeta_score(_labels, _predictions, beta=2, average="samples")
                    print(_labels)
                    print(_predictions)
                    #print(_f2_score)
                    f2_scores.append(_f2_score)
                    #labels_list.append(_labels)
                    #predictions_list.append(_predictions)
                    #name_list.append(name)
                    step += 1
                    
                
                f2_score_sum = 0
                for f2 in f2_scores:
                    f2_score_sum += f2

                avrg_f2_score = f2_score_sum / float(len(f2_scores))
                #print(avrg_f2_score)
                print('avg f2 score %.6f' % avrg_f2_score)
                
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)

def run_test():
    with tf.Graph().as_default():
        n_test = 61191
        images, _, images_filename = process_input.inputs_pipeline('sample_submission_v2_new.csv', BATCH_SIZE, None)
        logits = inference(images, is_training=False)
        final_tensor = get_final_tensor(logits)
        correct_prediction = round_prediction(final_tensor)

        saver = tf.train.Saver(tf.global_variables())
        
        with tf.Session() as sess:
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(tf.app.flags.FLAGS.log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return
        
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)
            
            try:
                num_iter = int(math.ceil(n_test / BATCH_SIZE))
                total_count = 0
                total_sample_count = num_iter * BATCH_SIZE
                step = 0
                total_images = []
                total_predictions = []
                
                while step < num_iter and not coord.should_stop():
                    image_names, predictions = sess.run([images_filename, correct_prediction])
                    total_images.extend(image_names)
                    total_predictions.extend(predictions)
                    #print(predictions)
                    step += 1
                    
                print (len(total_images))
                print (len(total_predictions))
                n = len(total_images)
                file_features = {}
                for i in range(n):
                    decode_file_image = total_images[i].decode()
                    file_name = decode_file_image[16:]
                    #print(file_name)
                    name, suffix = file_name.split('.')
                    features = []
                    for j in range(process_input.NUM_CLASSES):
                        if total_predictions[i][j] == 1:
                            features.append(process_input.LABELS_IDX_DICT[j])
                    features_str = " ".join(features)
                    if name in file_features:
                        print (name + "is already found")
                    else:
                        file_features[name] = features_str

                key_list = file_features.keys()
                key_list = sorted(key_list)

                with open('submit_kaggle_amazon.csv', 'w') as to_file:
                    to_file.write('image_name,tags\n')
                    for key in key_list:
                        new_line = key + ',' + file_features[key] + '\n'
                        to_file.write(new_line)
                to_file.close()
                
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)


def main():
    if tf.app.flags.FLAGS.is_test:
        run_test()
    else:
        if tf.app.flags.FLAGS.is_eval:
            evaluate()
        else:
            train()

if __name__ == '__main__':
    main()







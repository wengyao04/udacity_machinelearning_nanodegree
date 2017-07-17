import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy as np

IMAGE_SIZE = 32
NUM_CHANNELS = 3

LABELS_DICT = { 'haze': 0, 'primary': 1,  'agriculture': 2,  'clear': 3, 'water': 4,
                'habitation': 5,  'road': 6, 'cultivation': 7, 'slash_burn': 8, 'cloudy': 9,
                'partly_cloudy': 10, 'conventional_mine': 11, 'bare_ground': 12, 'artisinal_mine': 13,
                'blooming': 14, 'selective_logging': 15, 'blow_down': 16 }

LABELS_IDX_DICT = { 0: 'haze', 1: 'primary',  2: 'agriculture',  3: 'clear', 4: 'water',
                    5: 'habitation',  6: 'road', 7: 'cultivation', 8: 'slash_burn', 9: 'cloudy',
                    10: 'partly_cloudy', 11: 'conventional_mine', 12: 'bare_ground', 13: 'artisinal_mine',
                    14: 'blooming', 15: 'selective_logging', 16: 'blow_down'}

NUM_CLASSES = 17

def one_hot_encode(label_list):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    x_length = len(label_list)
    #print (x_length)
    ret = np.zeros(shape=(x_length, NUM_CLASSES))
    for i in range(x_length):
       label_str = label_list[i]
       label_str_list = label_str.split(' ')
       for one_label in label_str_list:
           #print (one_label)
           ret[i][LABELS_DICT[one_label]] = 1.
    return ret

def normalize(tensor):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    # min-max
    #amin = np.amin(x)
    #amax = np.amax(x)
    #return (x - amin) / (amax - amin)
    x = tf.div(tf.subtract(tensor, tf.reduce_min(tensor)), 
               tf.subtract(tf.reduce_max(tensor), tf.reduce_min(tensor)))
    return x

def read_labeled_image_list(image_list_file):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .cvs file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    for line in f:
        line_array = line[:-1].split(',')
        if len(line_array) == 2:
            filenames.append(line_array[0])
            labels.append(line_array[1])
        else:
            filenames.append(line_array[0])
            
    return filenames, labels

def preprocessing(image):
    resized_image = tf.image.resize_images(image, [IMAGE_SIZE, IMAGE_SIZE], method=0)
    resized_image.set_shape([IMAGE_SIZE, IMAGE_SIZE,3])
    return resized_image

def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    image_filename = input_queue[2]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=NUM_CHANNELS)
    processed_example = preprocessing(example)
    processed_label = label
    return processed_example, processed_label, image_filename

def inputs_pipeline(fname, batch_size, nepochs):

    # Reads pfathes of images together with their labels
    image_list, label_list = read_labeled_image_list(fname)
    label_list = one_hot_encode(label_list)
    
    images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
    labels = ops.convert_to_tensor(label_list, dtype=dtypes.float32)

    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels, image_list], num_epochs=nepochs, shuffle=False)

    image, label, image_filename = read_images_from_disk(input_queue)

    # Optional Preprocessing or Data Augmentation
    # tf.image implements most of the standard image augmentation
    image = normalize(image)
    
    # Optional Image and Label Batching
    image_batch, label_batch, image_filename_batch = tf.train.batch([image, label, image_filename], batch_size=batch_size)
    tf.summary.image('images', image_batch)
    return image_batch, label_batch, image_filename_batch

def test_pipeline(is_debug=False):
    image_batch, label_batch, image_filename_batch = inputs_pipeline('train_data_sample.csv', batch_size=1, nepochs=None)
    sess = tf.Session()
    
    with sess.as_default():
        # initialize the variables
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        # initialize the queue threads to start to shovel data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #print (label_batch.eval())
        #print (image_filename_batch.eval())

        for step in range(2):
            images, labels, image_names = sess.run([image_batch, label_batch, image_filename_batch])
            n = len(image_names)
            image_name_list = []
            for i in range(n):
                image_name_list.append(image_names[i].decode('utf-8'))
                image_name_str = ' '.join(image_name_list)

            print (str(step) + ' ' + image_name_str)
        
            for image in images:
                print(image)
                
            if is_debug:    
                for i in range(n):
                    #print (image_names[i])
                    print (labels[i])
                    label_list = []
                    for j in range(NUM_CLASSES):
                        if (int(labels[i][j])) == 1:
                            label_list.append(LABELS_IDX_DICT[int(j)])
                    label_str = ' '.join(label_list)
                    print (image_names[i].decode("utf-8") + ' ' + label_str)

        # stop our queue threads and properly close the session
        coord.request_stop()
        coord.join(threads)

        print ('Finish Test')
        sess.close()

def main():
    test_pipeline()
if __name__ == '__main__':
    main()

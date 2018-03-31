import cv2   
import tensorflow as tf  
import numpy as np  
import os  
#import matplotlib.pyplot as plt  
#import skimage.io as io  
#from skimage import transform  
  
#%%  
  
def get_files(file_dir):  
    ''''' 
    Args: 
        file_dir: file directory 
    Returns: 
        list of images and labels 
    '''  
    cats = []  
    label_cats = []  
    dogs = []  
    label_dogs = []  
    for file in os.listdir(file_dir):  
        name = file.split('.')  
        if name[0]=='cat':  
            cats.append(file_dir +'/'+ file)  
            label_cats.append(0)  
        else:  
            dogs.append(file_dir +'/'+ file)  
            label_dogs.append(1)  
    print('There are %d cats\nThere are %d dogs' %(len(cats), len(dogs)))  
      
    image_list = np.hstack((cats, dogs))  
    label_list = np.hstack((label_cats, label_dogs))  
      
    temp = np.array([image_list, label_list])  
    temp = temp.transpose()  
    np.random.shuffle(temp)  
      
    image_list = list(temp[:, 0])  
    label_list = list(temp[:, 1])  
    label_list = [int(i) for i in label_list]  
    
    return image_list, label_list  
  
  
#%%  
  
def int64_feature(value):  
  """Wrapper for inserting int64 features into Example proto."""  
  if not isinstance(value, list):  
    value = [value]  
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))  
  
def bytes_feature(value):  
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))  
  
#%%  
  
def convert_to_tfrecord(images, labels, save_dir, name):  
    '''''convert all images and labels to one tfrecord file. 
    Args: 
        images: list of image directories, string type 
        labels: list of labels, int type 
        save_dir: the directory to save tfrecord file, e.g.: '/home/folder1/' 
        name: the name of tfrecord file, string type, e.g.: 'train' 
    Return: 
        no return 
    Note: 
        converting needs some time, be patient... 
    '''  
      
    filename = (save_dir + name + '.tfrecords')  
    n_samples = len(labels)  
      
    if np.shape(images)[0] != n_samples:  
        raise ValueError('Images size %d does not match label size %d.' %(images.shape[0], n_samples))  
      
      
      
    # wait some time here, transforming need some time based on the size of your data.  
    writer = tf.python_io.TFRecordWriter(filename)  
    print('\nTransform start......')  
    for i in np.arange(0, n_samples):  
        try:  
            image = cv2.imread(images[i])    
            image = cv2.resize(image, (208, 208))    
            b,g,r = cv2.split(image)    
            rgb_image = cv2.merge([r,g,b])  # this is suitable    
#            image = rgb_image.astype(np.float32)   
#            image = io.imread(images[i]) # type(image) must be array!  #这边是两种读取图像的方法  
#            image = transform.resize(image, (208, 208))  
#            image = np.asarray(image)  
            image_raw =  rgb_image.tostring()  
            label = int(labels[i])  
            example = tf.train.Example(features=tf.train.Features(feature={  
                            'label':int64_feature(label),  
                            'image_raw': bytes_feature(image_raw)}))  
            writer.write(example.SerializeToString())  
        except IOError as e:  
            print('Could not read:', images[i])  
            print('error: %s' %e)  
            print('Skip it!\n')  
    writer.close()  
    print('Transform done!')  
      
  
#%%  
  
def read_and_decode(tfrecords_file, batch_size):  
    '''''read and decode tfrecord file, generate (image, label) batches 
    Args: 
        tfrecords_file: the directory of tfrecord file 
        batch_size: number of images in each batch 
    Returns: 
        image: 4D tensor - [batch_size, width, height, channel] 
        label: 1D tensor - [batch_size] 
    '''  
    # make an input queue from the tfrecord file  
    filename_queue = tf.train.string_input_producer([tfrecords_file])  
      
    reader = tf.TFRecordReader()  
    _, serialized_example = reader.read(filename_queue)  
    img_features = tf.parse_single_example(  
                                        serialized_example,  
                                        features={  
                                               'label': tf.FixedLenFeature([], tf.int64),  
                                               'image_raw': tf.FixedLenFeature([], tf.string),  
                                               })  
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)  
      
    ##########################################################  
    # you can put data augmentation here, I didn't use it  
    ##########################################################  
    # all the images of notMNIST are 28*28, you need to change the image size if you use other dataset.  
      
    image = tf.reshape(image, [208, 208,3])  
    label = tf.cast(img_features['label'], tf.float32)  
    
    image = tf.image.per_image_standardization(image)  
    image_batch, label_batch = tf.train.batch([image, label],  
                                                batch_size= batch_size,  
                                                num_threads= 64,   
                                                capacity = 2000)  
    
    return image_batch, tf.reshape(label_batch, [batch_size])  




# Generate image_lsist and label_list
#image_list, label_list = get_files("./train")
#print(len(image_list))
#print(len(label_list))
#print(image_list[0])
#print(label_list[0])

# Generate tfrcord file
#convert_to_tfrecord(image_list, label_list, './', 'train_cat_dog')

#read tfrecord file 这里只取了一次，因为没有iterator 的getnext()
image,label = read_and_decode('./train_cat_dog.tfrecords', batch_size=30)


with tf.Session() as sess:
    # 用Coordinator协同线程，并启动线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    image_show = sess.run(image)
    for i in range(30):
        cv2.imshow("test",image_show[i])
        cv2.waitKey() 
    coord.request_stop()
    coord.join(threads)
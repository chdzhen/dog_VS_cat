import model  
import numpy as np  
#from PIL import Image  
import tensorflow as tf  
import cv2
#import matplotlib.pyplot as plt  
#import input_data  

  
  
def get_one_img(img_dir): 
    image = cv2.imread(img_dir)
    #image = Image.open(img_dir)  
    #plt.imshow(image)  
    cv2.imshow("test",image)
    cv2.waitKey()
    image = cv2.resize(image,(208,208))
    image = np.array(image)  
    return image  
      
def evaluate_one_image():  
    '''''Test one image against the saved models and parameters 
    '''  
      
    # you need to change the directories to yours.  
    img_dir = 'C:/Users/zhen/Desktop/cat_test.jpg'  
    image_array = get_one_img(img_dir)  
  
      
    with tf.Graph().as_default():  
        BATCH_SIZE = 1  
        N_CLASSES = 2  
          
        image = tf.cast(image_array, tf.float32)  
        image = tf.image.per_image_standardization(image)  
        image = tf.reshape(image, [1, 208, 208, 3])  

        #x = tf.placeholder(tf.float32, shape=[1,208, 208, 3]) 
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)  
          
        logit = tf.nn.softmax(logit)  
          
         
          
        # you need to change the directories to yours.  
        logs_train_dir = './logdir/'   
                         
        saver = tf.train.Saver()  
          
        with tf.Session() as sess:  
              
            print("Reading checkpoints...")  
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)  
            if ckpt and ckpt.model_checkpoint_path:  
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]  
                saver.restore(sess, ckpt.model_checkpoint_path)  
                print('Loading success, global_step is %s' % global_step)  
            else:  
                print('No checkpoint file found')  
            #image_feed = sess.run(image)
            #prediction = sess.run(logit, feed_dict={x: image_feed})  
            prediction = sess.run(logit)  
            max_index = np.argmax(prediction)  
            if max_index==0:  
                print('This is a cat with possibility %.6f' %prediction[:, 0])  
            else:  
                print('This is a dog with possibility %.6f' %prediction[:, 1])  



evaluate_one_image()


'''
心得：由于image已经是一个tensor了，如果再重新建立一个placeholder，sess.run()的时候再进行feed，
就相当于image这个节点到下一个节点之间重新添加了一个节点。
这样从建立计算图的方面来说，是毫无意义的。
'''
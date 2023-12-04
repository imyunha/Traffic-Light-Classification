import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import sys


image_path = sys.argv[1]

# Loads label file, strips off carriage return
# tf.io.gfile.GFile() = open() i.e. file I/O function
label_lines = [line.rstrip() for line in tf.io.gfile.GFile("./retrained_labels.txt")]
# yellow, green, red


# Unpersists graph from file
# .pb(protobuf) file: binary file which consists of variables(weights) 
# and structure(Graph) of pre-trained model
with tf.io.gfile.GFile("./retrained_graph.pb", 'rb') as f:
    # GraphDef is protocol buffer that contains definition of TF graph
    graph_def = tf.compat.v1.GraphDef() # create new object
    graph_def.ParseFromString(f.read()) # read data from pb file and load to memory
    _ = tf.import_graph_def(graph_def, name='') # import loaded graph to current session

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    path = image_path
    file_handle = open("3.txt", mode='w')
    result = open("result.log", mode='w')

    import time
    start_time = time.time()
    for image in os.listdir(path):
        image_data = tf.gfile.GFile(path+image, 'rb').read()
        
        #import time
        #start_time = time.time()
        
        # possibility of each class by softmax
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        
        print(f"{image} : {predictions} {label_lines[predictions.argsort()[0][::-1][0]]}")
        result.write(f"{image} : {predictions} {label_lines[predictions.argsort()[0][::-1][0]]}\n")
        #file_handle.write(str(time.time()-start_time)+'\n')
    
    file_handle.close()
    end_time = time.time()
    
    # Sort to show labels of first prediction in order of confidence
    #top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    #for node_id in top_k:
    #    human_string = label_lines[node_id]
    #    score = predictions[0][node_id]
    #    print('%s (score = %.5f)' % (human_string, score))

    print("Inference Time: ", end_time - start_time, "sec")

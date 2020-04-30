

from styx_msgs.msg import TrafficLight

import rospy
import rospkg

import os
import sys

import tensorflow as tf
from io import StringIO
from utils import label_map_util
import time
import numpy as np

class TLClassifier(object):
    def __init__(self):
    
        #TODO load classifier
        rcnn_model =  'CarND-Capstone/ros/src/tl_detector/frozen_inference_graph_rcnn_10.pb'
        label_map_file = 'CarND-Capstone/ros/src/tl_detector/label_map.pbtxt'
        num_classes = 4

        label_map = label_map_util.load_labelmap(label_map_file)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        #self.image_np_deep = None TODO
        self.detection_graph = tf.Graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with self.detection_graph.as_default():
    
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(rcnn_model, 'rb') as fid:
        
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.sess = tf.Session(graph=self.detection_graph, config=config)


    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        #TODO implement light color prediction
        

        image_expanded = np.expand_dims(image, axis=0)
        with self.detection_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run([self.detection_boxes,self.detection_scores,self.detection_classes, self.num_detections], feed_dict={self.image_tensor: image_expanded})
        '''
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)
        self.current_light = TrafficLight.UNKNOWN
        score_threshold = 0.5
        for i in range(boxes.shape[0]):
            
            if scores is None or scores[i] > score_threshold:
        print("boxes shape : ", boxes[0].shape)
        class_name = self.category_index[classes[i]]['name']
        if class_name == 'Red':
            self.current_light = TrafficLight.RED
        elif class_name == 'Green':
            self.current_light = TrafficLight.GREEN
        elif class_name == 'Yellow':
            self.current_light = TrafficLight.YELLOW
            
        return self.current_light
        '''
        min_score_thresh = 0.5
        
        condition = scores > min_score_thresh
        detections_above_thresh = np.extract(condition, classes)
        unique_classes, counts = np.unique(detections_above_thresh, return_counts=True)
        most_probable_class = unique_classes[counts.argsort()[::-1]]
        tld_class = int(most_probable_class.item(0)) if len(most_probable_class) > 0 else 4

        if tld_class == 1:
            return TrafficLight.GREEN
        elif tld_class == 2:
            return TrafficLight.RED
        elif tld_class == 3:
            return TrafficLight.YELLOW
        else:
            return TrafficLight.UNKNOWN
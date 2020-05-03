from styx_msgs.msg import TrafficLight
import rospy
import numpy as np
import tensorflow as tf

graph_file_location = rospy.get_param("/tl_classifier_pb_file")

CONFIDENCE_THRESHOLD = 0.5

class TLClassifier(object):
    def __init__(self):
        
        # Load graph
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file_location, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        with tf.Session(graph=self.detection_graph) as sess:
            # Detection
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                                feed_dict={self.image_tensor: image_np})
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)
            
            # Filter boxes with a maximum confidence threshold
            idxs = []
            for i in range(len(classes)):
                if scores[i] >= CONFIDENCE_THRESHOLD:
                    idxs.append(i)
            boxes = boxes[idxs, ...]
            scores = scores[idxs, ...]
            classes = classes[idxs, ...]
            
            if classes.size != 0:
                if classes[0] == 1:
                    rospy.logwarn("[TL_CLASSIFIER] Green")
                    return TrafficLight.GREEN
                elif classes[0] == 2:
                    rospy.logwarn("[TL_CLASSIFIER] Red")
                    return TrafficLight.RED
                elif classes[0] == 3:
                    rospy.logwarn("[TL_CLASSIFIER] Yellow")
                    return TrafficLight.YELLOW

        rospy.logwarn("[TL_CLASSIFIER] Unknown")
        return TrafficLight.UNKNOWN
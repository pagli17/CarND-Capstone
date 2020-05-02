from styx_msgs.msg import TrafficLight
import rospy
import numpy as np
import tensorflow as tf

graph_file_location = rospy.get_param("/tl_classifier_pb_file")


def filter_boxes(min_score, boxes, scores, classes):
    """Return boxes with a confidence >= `min_score`"""
    n = len(classes)
    idxs = []
    for i in range(n):
        if scores[i] >= min_score:
            idxs.append(i)
    filtered_boxes = boxes[idxs, ...]
    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    return filtered_boxes, filtered_scores, filtered_classes

def load_graph(graph_file):
    """Loads a frozen inference graph"""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph


class TLClassifier(object):
    def __init__(self):
        #TODO load classifiier
        #graph_file_location = rospy.get_param("/tl_classifier_pb_file")
        
        self.detection_graph = load_graph(graph_file_location)
        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent the level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
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
            # Actual detection.
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                                feed_dict={self.image_tensor: image_np})

            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            confidence_threshold = 0.65
            # Filter boxes with a confidence score less than `confidence_cutoff`
            boxes, scores, classes = filter_boxes(confidence_threshold, boxes, scores, classes)
            if classes.size != 0:

                if classes[0] == 1:
                    rospy.logwarn("TL_Classifier State Green: {}".format(TrafficLight.GREEN))
                    return TrafficLight.GREEN
                elif classes[0] == 2:
                    rospy.logwarn("TL_Classifier State Red: {}".format(TrafficLight.RED))
                    return TrafficLight.RED
                elif classes[0] == 3:
                    rospy.logwarn("TL_Classifier State Yellow: {}".format(TrafficLight.YELLOW))
                    return TrafficLight.YELLOW

        #TODO implement light color prediction
        rospy.logwarn("TL_Classifier State Unknown: {}".format(TrafficLight.UNKNOWN))
        return TrafficLight.UNKNOWN
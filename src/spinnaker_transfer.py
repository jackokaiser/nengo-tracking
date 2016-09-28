import rospy
from robohead_controller import ServoController
from rospy.topics import Subscriber
from std_msgs.msg import Int64

self._pan_pub = Publisher("/head/pan", Int64, queue_size = 1)
self._tilt_pub = Publisher("/head/tilt", Int64, queue_size = 1)

def publish(arr):
   arr = np.mean(arr, 1)
   arr.reshape(5, 5)

   result = index(max(arr)) - (2, 2)

   self._pan_pub.publish(result[0])
   self._tilt_pub.publish(result[1])

#!/usr/bin/env python3
import roslib
import rospy
import cv2
import numpy as np
from std_msgs.msg import  Float64

def publisher():
    rospy.init_node('joint1_publisher', anonymous=True)
    pub1 = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size = 10)

    while not rospy.is_shutdown():
        t = rospy.get_time()
        joint1_pos = np.pi /2 * np.sin(np.pi/15 *t)
        pub1.publish(joint1_pos)


if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass

#!/usr/bin/env python3
import roslib
import rospy
import cv2
import numpy as np
from std_msgs.msg import  Float64

def publisher():
    rospy.init_node('joints_publisher', anonymous=True)
    pub2 = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size = 10)
    pub3 = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size = 10)
    pub4 = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size = 10)

    while not rospy.is_shutdown():
        t = rospy.get_time()
        joint2 = np.pi /2 * np.sin(np.pi/15 *t)
        joint3 = np.pi /2 * np.sin(np.pi/20 *t)
        joint4 = np.pi /2 * np.sin(np.pi/18 *t)

        pub2.publish(joint2)
        pub3.publish(joint3)
        pub4.publish(joint4)


if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass

#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError


class image_converter:

  # Defines publisher and subscriber
  def __init__(self):
    rospy.init_node('joint_angle_publisher', anonymous=True)
    self.joint2_pub = rospy.Publisher("joint_angle_2",Float64, queue_size = 1)
    self.joint3_pub = rospy.Publisher("joint_angle_3",Float64, queue_size = 1)
    self.joint4_pub = rospy.Publisher("joint_angle_4",Float64, queue_size = 1)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image, self.camera1_callback)
    self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw",Image, self.camera2_callback)




  # Recieve data, process it, and publish
  def camera1_callback(self,data):
    # Recieve the image
    try:
      image_camera1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    #code for camera 1 callback

    self.joint_angle2 = Float64()
    self.joint_angle2.data =
    self.joint_angle3 = Float64()
    self.joint_angle3.data =
    self.joint_angle4 = Float64()
    self.joint_angle4.data =

    # Publish the results
    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
      self.joints_pub.publish(self.joints)
    except CvBridgeError as e:
        print(e)



  def camera2_callback(self,data):
    # Recieve the image
    try:
      image_camera2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    #code for camera 2 callback

    self.joint_angle2 = Float64()
    self.joint_angle2.data =
    self.joint_angle3 = Float64()
    self.joint_angle3.data =
    self.joint_angle4 = Float64()
    self.joint_angle4.data =

    # Publish the results
    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
      self.joints_pub.publish(self.joints)
    except CvBridgeError as e:
        print(e)



  #detect the center of the joints in vertical and horizontal position
  def detect_red(self,image):
      mask = cv2.inRange(image, (0, 0, 100), (0, 0, 255))
      kernel = np.ones((5, 5), np.uint8)
      mask = cv2.dilate(mask, kernel, iterations=3)
      M = cv2.moments(mask)
      c_horizontal = int(M['m10'] / M['m00'])
      c_vertical = int(M['m01'] / M['m00'])
      return np.array([c_horizontal, c_vertical])


  def detect_blue(self,image):
      mask = cv2.inRange(image, (100, 0, 0), (255, 0, 0))
      kernel = np.ones((5, 5), np.uint8)
      mask = cv2.dilate(mask, kernel, iterations=3)
      M = cv2.moments(mask)
      c_horizontal = int(M['m10'] / M['m00'])
      c_vertical = int(M['m01'] / M['m00'])
      return np.array([c_horizontal, c_vertical])

  # Detecting the centre of the yellow circle
  def detect_yellow(self,image):
      mask = cv2.inRange(image, (0, 100, 100), (0, 255, 255))
      kernel = np.ones((5, 5), np.uint8)
      mask = cv2.dilate(mask, kernel, iterations=3)
      M = cv2.moments(mask)
      c_horizontal = int(M['m10'] / M['m00'])
      c_vertical = int(M['m01'] / M['m00'])
      return np.array([c_horizontal, c_vertical])

  def detect_green(self,image):
      mask = cv2.inRange(image, (0, 100, 0), (0, 255, 0))
      kernel = np.ones((5, 5), np.uint8)
      mask = cv2.dilate(mask, kernel, iterations=3)
      M = cv2.moments(mask)
      c_horizontal = int(M['m10'] / M['m00'])
      c_vertical = int(M['m01'] / M['m00'])
      return np.array([c_horizontal, c_vertical])


  # Calculate the conversion from pixel to meter
  def pixel2meter(self,image):
      circle1Pos = self.detect_green(image)
      circle2Pos = self.detect_yellow(image)
      dist = np.sum((circle1Pos - circle2Pos)**2)
      return 4 / np.sqrt(dist)


      print(e)

  # Calculate the relevant joint angles from the image
  def detect_joint_angles_1(self,image):
    a = self.pixel2meter(image)
    # Obtain the centre of each coloured blob
    greenPos = a * self.detect_green(image)
    yellowPos = a * self.detect_yellow(image)
    bluePos = a * self.detect_blue(image)
    redPos = a * self.detect_red(image)
    # Solve using trigonometry
    ja3 = np.arctan2(bluePos[0] - yellowPos[0], bluePos[1] - yellowPos[1])
    return np.array([ja3])
    # ja1 = np.arctan2(greenPos[0]- yellowPos[0], greenPos[1] - yellowPos[1])
    # ja2 = np.arctan2(yellowPos[0]-bluePos[0], yellowPos[1]-bluePos[1]) - ja1
    # ja3 = np.arctan2(redPos[0]-bluePos[0], redPos[1]-bluePos[1]) - ja2 - ja1
    # return np.array([ja1, ja2, ja3])

  def detect_joint_angles_2(self,image):
    a = self.pixel2meter(image)
    # Obtain the centre of each coloured blob
    greenPos = a * self.detect_green(image)
    yellowPos = a * self.detect_yellow(image)
    bluePos = a * self.detect_blue(image)
    redPos = a * self.detect_red(image)
    # Solve using trigonometry
    ja2 = np.arctan2(bluePos[0] - yellowPos[0], bluePos[1] - yellowPos[1])
    ja4 = np.arctan2(redPos[0] - bluePos[0], redPos[1] - bluePos[1])
    return np.array([ja2, ja4])
    # ja1 = np.arctan2(greenPos[0]- yellowPos[0], greenPos[1] - yellowPos[1])
    # ja2 = np.arctan2(yellowPos[0]-bluePos[0], yellowPos[1]-bluePos[1]) - ja1
    # ja3 = np.arctan2(redPos[0]-bluePos[0], redPos[1]-bluePos[1]) - ja2 - ja1
    # return np.array([ja1, ja2, ja3])

# call the class
def main(args):
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)

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

redPrev1 = np.array([0, 0])
redPrev2 = np.array([0, 0])

bluePrev1 = np.array([0, 0])
bluePrev2 = np.array([0, 0])

class image_converter:

  # Defines publisher and subscriber
  def __init__(self):
    rospy.init_node('joint_angle_publisher', anonymous=True)
    self.joint2_pub = rospy.Publisher("joint_angle_2",Float64, queue_size = 1)
    self.joint3_pub = rospy.Publisher("joint_angle_3",Float64, queue_size = 1)
    self.joint4_pub = rospy.Publisher("joint_angle_4",Float64, queue_size = 1)
    # rospy.init_node('image_processing', anonymous=True)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()
    # initialize a publisher to send messages to a topic named image_topic
    self.image_pub = rospy.Publisher("image_topic", Image, queue_size=1)
    # initialize a publisher to send joints' angular position to a topic called joints_pos
    self.joints_pub = rospy.Publisher("joints_pos", Float64MultiArray, queue_size=10)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image, self.camera1_callback)
    self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw",Image, self.camera2_callback)

  #detect the center of the joints in vertical and horizontal position
  def detect_red(self,image):
      mask = cv2.inRange(image, (0, 0, 100), (0, 0, 255))
      kernel = np.ones((5, 5), np.uint8)
      mask = cv2.dilate(mask, kernel, iterations=3)
      M = cv2.moments(mask)
      if M['m00'] == 0:
          return np.array([-1, -1])
      else:
          c_horizontal = int(M['m10'] / M['m00'])
          c_vertical = int(M['m01'] / M['m00'])
          return np.array([c_horizontal, c_vertical])


  def detect_blue(self,image):
      mask = cv2.inRange(image, (100, 0, 0), (255, 0, 0))
      kernel = np.ones((5, 5), np.uint8)
      mask = cv2.dilate(mask, kernel, iterations=3)
      M = cv2.moments(mask)
      if M['m00'] == 0:
          return np.array([-1, -1])
      else:
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
  def detect_joint_angles(self):

    # meter conversion constants
    a = self.pixel2meter(self.image_camera1)
    b = self.pixel2meter(self.image_camera2)

    # Obtain the centre of each coloured blob
    greenPos1 = a * self.detect_green(self.image_camera1)
    greenPos2 = a * self.detect_green(self.image_camera2)

    yellowPos1 = a * self.detect_yellow(self.image_camera1)
    yellowPos2 = a * self.detect_yellow(self.image_camera2)

    bluePos1 = a * self.detect_blue(self.image_camera1)
    bluePos2 = a * self.detect_blue(self.image_camera2)

    # blue blocked
    if bluePos1[0] == -1:
        bluePos1 = bluePrev1
    if bluePos2[0] == -1:
        bluePos2 = bluePrev2

    for i in range(len(bluePos1)):
        bluePrev1[i] = bluePos1[i]
        bluePrev2[i] = bluePos2[i]

    redPos1 = a * self.detect_red(self.image_camera1)
    redPos2 = a * self.detect_red(self.image_camera1)

    # red blocked
    if redPos1[0] == -1:
        redPos1 = redPrev1
    if redPos2[0] == -1:
        redPos2 = redPrev2

    for i in range(len(redPos1)):
        redPrev1[i] = redPos1[i]
        redPrev2[i] = redPos2[i]

    greenPos = np.array([greenPos2[0], greenPos1[0], (greenPos1[1] + greenPos2[1]) / 2])
    yellowPos = np.array([yellowPos2[0], yellowPos1[0], (yellowPos1[1] + yellowPos2[1]) / 2])

    bluePos = np.array([bluePos2[0] - greenPos[0], bluePos1[0] - greenPos[1], 0])
    redPos = np.array([redPos2[0] - greenPos[0], redPos1[0] - greenPos[1], 0])

    if yellowPos[2] > bluePos1[1] and yellowPos[2] > bluePos2[1]:
        bluePos[2] = greenPos[2] - ((bluePos1[1] + bluePos2[1]) / 2)
    else:
        bluePos[2] = greenPos[2] - yellowPos[2]

    if yellowPos[2] > redPos1[1] and yellowPos[2] > redPos2[1]:
        redPos[2] = greenPos[2] - ((redPos1[1] + redPos2[1]) / 2)
    else:
        redPos[2] = greenPos[2] - yellowPos[2]

    # find joint arm vectors
    yellowBlue = bluePos - yellowPos
    blueRed = redPos - bluePos

    print(yellowBlue)
    print(blueRed)

    print(np.linalg.norm(yellowBlue))
    print(np.linalg.norm(blueRed))


    # find x-axis after joint 2 rotation
    newX = np.cross(yellowBlue, np.array([0, 1, 0]))

    # find angle of joints
    joint2 = np.arccos((np.dot(newX, np.array([1, 0, 0]))) / (np.linalg.norm(newX) + np.linalg.norm(np.array([1, 0, 0]))))

    # if joint2 > np.pi/2:
    #     joint2 = np.pi - joint2
    # elif joint2 < -np.pi/2:
    #     joint2 = joint2 - np.pi
    #
    # if bluePos[0] < 0:
    #     joint2 = -joint2

    joint3 = np.arccos((np.dot(yellowBlue, np.array([0, 1, 0]))) / (np.linalg.norm(yellowBlue) + np.linalg.norm(np.array([0, 1, 0])))) - np.pi/2

    # if joint3 > np.pi/2:
    #     joint3 = np.pi - joint3
    # elif joint3 < -np.pi/2:
    #     joint3 = joint3 - np.pi

    joint4 = np.arccos((np.dot(yellowBlue, blueRed)) / (np.linalg.norm(yellowBlue) + np.linalg.norm(blueRed)))

    return [joint2, joint3, joint4]

  # Recieve data, process it, and publish
  def camera1_callback(self,data):
    # Recieve the image
    try:
      self.image_camera1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    # code for camera 1 callback
    try:
        joints = self.detect_joint_angles()
        self.joint_angle2 = Float64()
        self.joint_angle2.data = joints[0]
        self.joint_angle3 = Float64()
        self.joint_angle3.data = joints[1]
        self.joint_angle4 = Float64()
        self.joint_angle4.data = joints[2]

        self.joint2_pub.publish(self.joint_angle2.data)
        self.joint3_pub.publish(self.joint_angle3.data)
        self.joint4_pub.publish(self.joint_angle4.data)
    except CvBridgeError as e:
        print(e)

    # Publish the results
    # try:
    #     self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    #     self.joints_pub.publish(self.joints)
    # except CvBridgeError as e:
    #     print(e)

  def camera2_callback(self,data):
    # Recieve the image
    try:
      self.image_camera2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    #code for camera 2 callback

    # self.joint_angle2 = Float64()
    # self.joint_angle2.data =
    # self.joint_angle3 = Float64()
    # self.joint_angle3.data =
    # self.joint_angle4 = Float64()
    # self.joint_angle4.data =

    # Publish the results
    # try:
    #   self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    #   self.joints_pub.publish(self.joints)
    # except CvBridgeError as e:
    #     print(e)

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

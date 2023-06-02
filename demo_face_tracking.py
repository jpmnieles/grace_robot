import os
import time
import cv2
import dlib
import numpy as np

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class VisualizeSaccade(object):


    def __init__(self):
        rospy.init_node("eye_camera_subscriber")
        self.bridge = CvBridge()
        self.disp_img = None
        self.cropped_img = None
        self.bool_crop = False
        self.left_eye_sub = rospy.Subscriber('/eye_camera/left_eye/image_raw', Image, self._capture_left_image)
        # self.right_eye_sub = rospy.Subscriber('/eye_camera/right_eye/image_raw', Image, self._capture_right_image)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(os.path.join(os.getcwd(),'pretrained','shape_predictor_68_face_landmarks.dat'))
        time.sleep(1)
        self.main()

    def _capture_left_image(self, msg):
        try:
            self.left_eye_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as error:
            print(error)
        self.disp_img = self.process_img(self.left_eye_img)
    
    def _ctr_cross_img(self, img):
        img = cv2.line(img, (315, 0), (315, 480), (0,255,0))
        img = cv2.line(img, (0, 202), (640, 202), (0,255,0))
        img = cv2.drawMarker(img, (315,202), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
        img = cv2.rectangle(img, (0,449), (639,479), color=(0,0,0), thickness=-1)
        return img
    
    def _display_target(self, delta_x, delta_y, img):
        abs_x = 314.69441889 + delta_x
        abs_y = 201.68845842 - delta_y
        disp_img = cv2.drawMarker(img, (round(abs_x),round(abs_y)), color=(255, 0, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=13, thickness=2)
        return disp_img
    
    # def _capture_right_image(self, msg):
    #     try:
    #         self.right_eye_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    #     except CvBridgeError as error:
    #         print(error)

    def center_crop(self, img, dim):
        """Returns center cropped image
        Args:
        img: image to be center cropped
        dim: dimensions (width, height) to be cropped
        """
        width, height = img.shape[1], img.shape[0]

        # process crop width and height for max available dimension
        crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
        crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
        mid_x, mid_y = 315, 202
        cw2, ch2 = int(crop_width/2), int(crop_height/2) 
        crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
        return crop_img

    def scale_image(self, img, factor=1):
        """Returns resize image by scale factor.
        This helps to retain resolution ratio while resizing.
        Args:
        img: image to be scaled
        factor: scale factor to resize
        """
        return cv2.resize(img,(int(img.shape[1]*factor), int(img.shape[0]*factor)))

    def process_img(self, base_img):
        # Capture a frame
        img = base_img.copy()
          
        # Detection       
        gray = cv2.cvtColor(base_img.copy(), cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        detections = self.detector(gray, 0)

        # Loop over each detected face
        for detection in detections:
            # Get the facial landmarks for the detected face
            landmarks = self.predictor(gray, detection)

            # Draw a bounding box around the detected face
            x1 = detection.left()
            y1 = detection.top()
            x2 = detection.right()
            y2 = detection.bottom()
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        img = self._ctr_cross_img(img)

        # Identifying Target
        if len(detections) > 0:
            landmarks = self.predictor(gray, detections[0])
            x_target = landmarks.part(30).x
            y_target = landmarks.part(30).y
            delta_x = x_target-317.13846547
            delta_y =  219.22972847 - y_target

            img = self._display_target(delta_x, delta_y, img)
            img = cv2.putText(img, f'delta_x={delta_x:.4f} (px), delta_y={delta_y:.4f} (px)', (170,470), cv2.FONT_HERSHEY_SIMPLEX,
                   fontScale=0.5, color=(0,255,0), thickness=1)
            
            if self.bool_crop:
                after_cropped_img = img.copy()
                after_cropped_img = self.center_crop(after_cropped_img, dim=(224,224))
                after_cropped_img = self.scale_image(after_cropped_img, factor=2.2)
                after_cropped_img = cv2.putText(after_cropped_img, f'AFTER SACCADE', (160,25), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.75, color=(0,255,0), thickness=2)
                self.after_cropped_img = cv2.putText(after_cropped_img, f'delta_x={delta_x:.4f} (px), delta_y={delta_y:.4f} (px)', (25,50), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.60, color=(0,255,0), thickness=2)
                self.cropped_img = np.concatenate((self.before_cropped_img, self.after_cropped_img), axis=1)
                self.bool_crop = False

            if abs(delta_x) > 50 or abs(delta_y) > 50:
                before_cropped_img = img.copy()
                before_cropped_img = self.center_crop(before_cropped_img, dim=(224,224))
                before_cropped_img = self.scale_image(before_cropped_img, factor=2.2)
                before_cropped_img = cv2.putText(before_cropped_img, f'BEFORE SACCADE', (150,25), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.75, color=(0,255,0), thickness=2)
                self.before_cropped_img = cv2.putText(before_cropped_img, f'delta_x={delta_x:.4f} (px), delta_y={delta_y:.4f} (px)', (25,50), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.60, color=(0,255,0), thickness=2)
                self.bool_crop = True
    
        return img

    def main(self):
        rate = rospy.Rate(30) # 30hz
        while not rospy.is_shutdown():
            cv2.imshow('Left Eye', self.disp_img)
            if self.cropped_img is not None:
                cv2.imshow('Zoomed Image', self.cropped_img)
            key = cv2.waitKey(1)

            if key == 27:
                break

            rate.sleep()
        
        cv2.destroyAllWindows()
    

if __name__ == '__main__':
    eye_cam = VisualizeSaccade()

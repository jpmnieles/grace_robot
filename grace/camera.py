import os
import cv2
from datetime import datetime


class BaseCapture(object):
    

    def __init__(self) -> None:
        self._frame = None


    @property
    def frame(self):
        raise(NotImplementedError)



class LeftEyeCapture(BaseCapture):


    ID =  "/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0"


    def __init__(self, id=None) -> None:
        super().__init__()
        if id == None:
            id = self.ID
        cam_path = os.path.realpath(id)
        self.cap = cv2.VideoCapture(cam_path)


    @property
    def frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print('Left Eye Camera: No Captured Frame')        
        return frame


    def __exit__(self):
        self.cap.release()


class RightEyeCapture(BaseCapture):


    ID =  "/dev/v4l/by-id/usb-Generic_USB_Camera_20221013-OOV2740-13-0000-video-index0"



    def __init__(self, id=None) -> None:
        super().__init__()
        if id == None:
            id = self.ID
        cam_path = os.path.realpath(id)
        self.cap = cv2.VideoCapture(cam_path)


    @property
    def frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print('Right Eye Camera: No Captured Frame')        
        return frame


    def __exit__(self):
        self.cap.release()


if __name__ == "__main__":
    
    left_cam = LeftEyeCapture()
    right_cam = RightEyeCapture()

    while(1):
        try:
            cv2.imshow("Left Camera", left_cam.frame)
        except:
            pass
        
        try:
            cv2.imshow("Right Camera", right_cam.frame)
        except:
            pass

        key = cv2.waitKey(1)

        if key == 108:  # letter l
            """Save left eye camera image. Press 'l' key
            """
            frame = left_cam.frame
            print("Left Eye Camera Shape:", frame.shape)
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f") 
            fn_str = 'results/' + date_str + '_left_eye.png'
            cv2.imwrite(fn_str, frame)
            print("Saving Left Eye Camera Image to: ", fn_str)
        elif key == 114:  # letter r
            """Save right eye camera image. Press 'r' key
            """
            frame = right_cam.frame
            print("Right Eye Camera:", frame.shape)
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f") 
            fn_str = 'results/' + date_str + '_right_eye.png'
            cv2.imwrite(fn_str, frame)
            print("Saving Right Eye Camera Image to: ", fn_str)
        elif key == 115:  # letter s
            """Save both left & right eye camera image. Press 's' key
            """
            l_frame = left_cam.frame
            r_frame = right_cam.frame
            print("Left Eye Camera Shape:", l_frame.shape)
            print("Right Eye Camera:", r_frame.shape)
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f") 
            l_fn_str = 'results/' + date_str + '_left_eye.png'
            r_fn_str = 'results/' + date_str + '_right_eye.png'
            cv2.imwrite(l_fn_str, l_frame)
            cv2.imwrite(r_fn_str, r_frame)
            print("Saving Left Eye Camera Image to: ", l_fn_str)
            print("Saving Right Eye Camera Image to: ", r_fn_str)
        elif key == 27:  # Esc
            """Execute end of program. Press 'esc' key to escape program
            """
            del left_cam
            del right_cam
            break

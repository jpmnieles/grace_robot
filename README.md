# Grace Robot
Repository for Grace Robot Gaze Controllers

## Setup

1. Every terminal must be connected to the IP of the ros master
~~~
source IP_Setup_Local.bash
~~~
2. Launch the video stream node for the left and right eye cameras
~~~
roslaunch launch/video_steam_opencv-grace_nitro.launch
~~~
3. Run the robot state publisher node
~~~
source robot_state_publisher.bash
~~~
4. Run the command joint state publisher node
~~~
python joint_state_publisher.py
~~~
5. Run the output display for the gaze script
~~~
rosrun image_view image_view image:=/output_display1
~~~
6. Run the gaze script or custom script
~~~
python -m grace.tf_gaze
~~~

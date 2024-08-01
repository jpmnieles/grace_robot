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
3. Run the output display for the gaze script
~~~
rosrun image_view image_view image:=/output_display1
~~~
4. Run the gaze script or custom script
~~~
python scripts/headeyes_pantilt_baseline_sweep_charuco_240610.py
~~~


## Key Press Calback
1. sudo su
2. source /home/jaynieles/dev/aec/venv/bin/activate
3. source IP_Setup_Local.bash
4. python key_press.py
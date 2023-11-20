export ROS_MASTER_URI=http://192.168.99.10:11311
if [ "$HOSTNAME" = EE4E170 ]; then
    export ROS_IP=192.168.99.243
elif [ "$HOSTNAME" = nitro ]; then
    export ROS_IP=192.168.99.242
else
    export ROS_IP=192.168.99.243
fi
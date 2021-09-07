# autonomous_turbine_inspection

ROS package for performing autonomous wind turbine inspections. The package is build for ROS melodic on Ubuntu 18.04

# Setup
First ensure ROS Melodic is installed: ```sudo apt-get install ros-melodic-desktop-full```

The package builds on top of PX4 and MAVROS packages, which needs to be installed and configured.

## PX4 setup

Install dependencies:
```
sudo apt-get install astyle build-essential ccache clang clang-tidy cmake cppcheck doxygen file g++ gcc gdb git lcov make ninja-build python3 python3-dev python3-pip python3-setuptools python3-wheel rsync shellcheck unzip xsltproc zip libeigen3-dev libopencv-dev libroscpp-dev protobuf-compiler python-pip python3-pip ninja-build gstreamer1.0-plugins-bad gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-ugly libgstreamer-plugins-base1.0-dev libgstrtspserver-1.0-dev xvfb
```
```
pip install --user argparse cerberus empy jinja2 numpy packaging pandas psutil pygments pyros-genmsg pyserial pyulog pyyaml setuptools six toml wheel rosdep
```
```
pip3 install --user --upgrade empy jinja2 numpy packaging pyros-genmsg toml pyyaml pymavlink
```
For convenience when working with ROS also install the Python-based catkin tools:
```
sudo apt install python-catkin-tools
```
Then clone the PX4 firmware from Github and checkout to version 1.12.1 (stable release):
```
git clone https://github.com/PX4/PX4-Autopilot
cd ~/PX4-Autopilot
git checkout v1.12.1
git submodule update --init --recursive
```
Build the PX4-Autopilot SITL firmware and run the Gazebo environment
```
cd ~/PX4-Autopilot
DONT_RUN=1 make px4_sitl_default gazebo
```
Add the following lines to '.bashrc'
```
source /home/$USER/PX4-Autopilot/Tools/setup_gazebo.bash /home/$USER/PX4-Autopilot /home/$USER/PX4-Autopilot/build/px4_sitl_default
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:/home/$USER/PX4-Autopilot
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:/home/$USER/PX4-Autopilot/Tools/sitl_gazebo
```

## Workspace/MAVROS setup
The workspace has to be setup properly in order to link correctly to MAVROS and PX4.\
First, make sure 'wstool', 'rosinstall' and 'catkin_tools' are availble. Also install MAVROS.
```
sudo apt-get install python-catkin-tools python-rosinstall-generator -y
sudo apt install ros-noetic-mavros ros-noetic-mavros-extras -y
```
Then create and init a catkin workspace
```
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin init
wstool init src
```
Install MAVLink
```
rosinstall_generator --rosdistro melodic mavlink | tee /tmp/mavros.rosinstall
```
Install MAVROS from source using either released/stable version
```
rosinstall_generator --upstream mavros | tee -a /tmp/mavros.rosinstall
```
Create workspace & deps
```
wstool merge -t src /tmp/mavros.rosinstall
wstool update -t src -j4
rosdep install --from-paths src --ignore-src -y
```
Install GeographicLib datasets
```
sudo ./src/mavros/mavros/scripts/install_geographiclib_datasets.sh
```
We are now ready to build
```
catkin build
```
And remember to source the workspace..
```
source devel/setup.bash
```

## Configure this package
The workspace should now be set up correctly and this package can be cloned:
```
cd ./src
git clone https://github.com/MagnusAagaard/autonomous_turbine_inspection.git
cd ..
catkin build
source devel/setup.bash
```

### Launching the PX4 SITL with ROS wrapper
To run the simulation wrapped in ROS:
```
roslaunch px4 posix_sitl.launch
```
And with MAVROS:
```
roslaunch px4 mavros_posix_sitl.launch
```

# Package usage
The usage of this package is described here..

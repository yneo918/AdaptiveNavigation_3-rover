import os

from ament_index_python import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import TextSubstitution
from launch_ros.actions import Node
from launch_ros.actions import PushRosNamespace

from launch_ros.substitutions import FindPackageShare



def generate_launch_description():
    ld = LaunchDescription()
    
    # Nodes
    run_joy_node = Node(
        package="joy",
        executable="joy_node",
    )

    joy_to_cmd_vel = Node(
        package="teleop_core",
        executable="joywithgui3",
        parameters=["pioneer_base/teleop_core/config/joy-assign.yaml"],
    )

    demux = Node(
        package="teleop_core",
        executable="cmd_demux",
        parameters=["pioneer_base/teleop_core/config/demux.yaml"],
    )

    

    ld.add_action(run_joy_node)
    ld.add_action(joy_to_cmd_vel)
    ld.add_action(demux)

    return ld


from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='apple_care_robot',
            executable='motion_planner_node',
            name='motion_planner_node',
            output='screen',
        ),
        Node(
            package='apple_care_robot',
            executable='robot_controller_node',
            name='robot_controller_node',
            output='screen',
        ),
    ])

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # OnRobot 그리퍼는 onrobot_rg_control ROS2 서비스 노드에 의존하지 않고
    # openclose.py가 Modbus/TCP로 직접 제어하므로, 여기서 별도 그리퍼 서버를
    # 띄울 필요가 없습니다.
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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # 실물 하드웨어에 연결할 때만 'modbus'로 덮어써서 사용하세요.
    # (시뮬레이션에서 modbus로 두면 그리퍼 서버가 실제 IP 연결을 시도하다 죽어서
    #  /onrobot/sendCommand 서비스가 응답하지 않는 문제가 발생합니다.)
    gripper_control_arg = DeclareLaunchArgument(
        'gripper_control',
        default_value='isaac',
        description="OnRobot 그리퍼 통신 모드: 'isaac'(시뮬레이션) 또는 'modbus'(실물 하드웨어)",
    )

    onrobot_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('onrobot_rg_control'),
                'launch',
                'bringup.launch.py',
            )
        ),
        launch_arguments={
            'control': LaunchConfiguration('gripper_control'),
        }.items(),
    )

    return LaunchDescription([
        gripper_control_arg,
        onrobot_bringup,
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

"""
basket_camera.launch.py
========================
바스켓(B1~B4) 조망용 세컨 RealSense 카메라를 camera_name=camera2 네임스페이스로
띄우고, basket_camera 노드를 그 위에서 실행한다.

serial_no는 launch argument로 비워둔 채 제공한다 - 실제 카메라가 물리적으로
연결되고 시리얼 넘버가 확정되면 실행 시 `serial_no:=<번호>`로 넘기거나 이
launch 파일의 default_value를 채워 넣으면 된다 (여러 대의 RealSense가 같은
USB 버스에 연결됐을 때 realsense2_camera가 올바른 장치를 고르게 하는 값).

기존 파이프(get_apple_status용) 카메라는 realsense2_camera 기본
네임스페이스(/camera/camera/...)를 그대로 쓰므로, 여기서는 이름 충돌을
피하기 위해 camera_name/camera_namespace를 모두 "camera2"로 지정해
/camera2/camera/color/image_raw 등으로 토픽을 분리한다.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    serial_no_arg = DeclareLaunchArgument(
        'serial_no',
        default_value='',
        description=(
            '바스켓 조망용 세컨 RealSense의 시리얼 넘버. 카메라가 한 대만 '
            '연결되어 있으면 비워둬도 되지만, 픽 카메라와 동시에 연결된 상태라면 '
            '반드시 지정해야 realsense2_camera가 올바른 장치를 연다.'
        ),
    )

    realsense_launch_path = os.path.join(
        get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py'
    )
    basket_realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_launch_path),
        launch_arguments={
            'camera_name': 'camera2',
            'camera_namespace': 'camera2',
            'serial_no': LaunchConfiguration('serial_no'),
            'align_depth.enable': 'false',
        }.items(),
    )

    basket_camera_node = Node(
        package='obj_detection',
        executable='basket_camera',
        name='basket_detection_node',
        parameters=[{'camera_topic_prefix': '/camera2/camera'}],
        output='screen',
    )

    return LaunchDescription([
        serial_no_arg,
        basket_realsense,
        basket_camera_node,
    ])

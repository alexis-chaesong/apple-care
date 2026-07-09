"""
basket_camera.launch.py
========================
바스켓(B1~B4) 조망용 웹캠으로 basket_camera 노드를 실행한다.

이 카메라는 RealSense가 아니라 일반 V4L2 웹캠(예: 로지텍 C270)이라 별도의
카메라 드라이버 노드가 필요 없다 - basket_camera.py가 device_index 파라미터로
cv2.VideoCapture를 직접 열어서 프레임을 읽는다. device_index는
`v4l2-ctl --list-devices`로 확인한 /dev/videoN의 N (예: 39)을 넘기면 된다.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    device_index_arg = DeclareLaunchArgument(
        'device_index',
        default_value='0',
        description=(
            '바스켓 조망용 웹캠의 /dev/videoN 인덱스. '
            "'v4l2-ctl --list-devices'로 실제 장치와 인덱스를 확인한 뒤 넘길 것 "
            "(예: 로지텍 C270이 /dev/video39면 device_index:=39)."
        ),
    )

    frame_width_arg = DeclareLaunchArgument(
        'frame_width', default_value='1280',
        description="캡처 해상도 너비. 카메라가 지원 안 하는 값이면 검은 화면만 나오니 "
                    "'v4l2-ctl --list-formats-ext'로 지원 해상도를 먼저 확인할 것.",
    )
    frame_height_arg = DeclareLaunchArgument(
        'frame_height', default_value='720',
        description="캡처 해상도 높이 (frame_width 설명 참고).",
    )

    basket_camera_node = Node(
        package='obj_detection',
        executable='basket_camera',
        name='basket_detection_node',
        parameters=[{
            'device_index': ParameterValue(LaunchConfiguration('device_index'), value_type=int),
            'frame_width': ParameterValue(LaunchConfiguration('frame_width'), value_type=int),
            'frame_height': ParameterValue(LaunchConfiguration('frame_height'), value_type=int),
        }],
        output='screen',
    )

    return LaunchDescription([
        device_index_arg,
        frame_width_arg,
        frame_height_arg,
        basket_camera_node,
    ])

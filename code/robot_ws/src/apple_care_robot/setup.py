from setuptools import setup

package_name = 'apple_care_robot'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/apple_care.launch.py']),
        ('share/' + package_name + '/resource', ['resource/T_gripper2camera.npy']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='you@example.com',
    description='Apple_Care - Motion Planner & Robot Controller nodes',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'motion_planner_node = apple_care_robot.motion_planner_node:main',
            'robot_controller_node = apple_care_robot.robot_controller_node:main',
            # 실제 운영 진입점은 이 하나뿐. apple_sorting_cycle.py는 이제 main()이
            # 없는 공용 함수 모듈(pick_apple 등)이라 콘솔 스크립트로 등록하지 않음.
            'box_sequence_test = apple_care_robot.box_sequence_test:main',
        ],
    },
)

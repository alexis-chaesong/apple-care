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
            'box_sequence_test = apple_care_robot.box_sequence_test:main',
            'apple_sorting_cycle = apple_care_robot.apple_sorting_cycle:main',
        ],
    },
)

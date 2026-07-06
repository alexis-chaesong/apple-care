import glob
from setuptools import find_packages, setup

package_name = 'voice_processing'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/resource',
            glob.glob('resource/.env*') + glob.glob('resource/*.tflite')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='pfr0213@gmail.com',
    description='웨이크워드+STT+키워드추출 음성 인식과, 미확인 물체 감지 시 질문을 트리거하는 노드',
    license='apache2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'get_keyword = voice_processing.get_keyword:main',
            'unknown_object_watcher = voice_processing.unknown_object_watcher:main',
        ],
    },
)

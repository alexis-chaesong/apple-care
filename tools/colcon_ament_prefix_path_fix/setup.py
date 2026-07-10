from setuptools import find_packages, setup

setup(
    name='colcon-ament-prefix-path-fix',
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    zip_safe=True,
    author='apple-care',
    description=(
        'Local colcon-core environment extension that fills the missing '
        'ament_prefix_path hook for ament_cmake packages (colcon-ros 0.5.0 '
        'gap workaround). See BEFORE_AFTER_INTEGRATION.md.'
    ),
    license='Apache License, Version 2.0',
    entry_points={
        'colcon_core.environment': [
            'ament_prefix_path = '
            'colcon_ament_prefix_path_fix.environment.ament_prefix_path:'
            'AmentPrefixPathEnvironment',
        ],
    },
)

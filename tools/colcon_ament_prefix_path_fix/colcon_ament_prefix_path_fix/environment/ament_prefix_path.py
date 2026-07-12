from pathlib import Path

from colcon_core.environment import EnvironmentExtensionPoint
from colcon_core.plugin_system import satisfies_version
from colcon_core.shell import create_environment_hook


class AmentPrefixPathEnvironment(EnvironmentExtensionPoint):
    """
    Prepend AMENT_PREFIX_PATH for packages that install an ament index marker.

    Local workaround for a gap in colcon-ros 0.5.0: its
    ``AmentCmakeBuildTask`` never registers an ``ament_prefix_path`` hook
    (unlike ``AmentPythonBuildTask``, which creates one inline), and no
    ``colcon_core.environment`` extension fills the gap on this system. As a
    result ``ament_cmake``/``rosidl_generate_interfaces`` packages never
    contribute to ``AMENT_PREFIX_PATH`` in the workspace-level
    ``install/setup.bash``, even though CMake/ament_cmake_core itself
    correctly generates a (dead, unreferenced)
    ``share/<pkg>/environment/ament_prefix_path.sh`` for them.

    This extension mirrors what colcon-ros already does for ament_python
    packages, generalized to any package that installs the ament resource
    index marker ``share/ament_index/resource_index/packages/<pkg_name>``.
    """

    def __init__(self):  # noqa: D107
        super().__init__()
        satisfies_version(
            EnvironmentExtensionPoint.EXTENSION_POINT_VERSION, '^1.0')

    def create_environment_hooks(self, prefix_path, pkg_name):  # noqa: D102
        prefix_path = Path(prefix_path)
        marker = (
            prefix_path / 'share' / 'ament_index' / 'resource_index' /
            'packages' / pkg_name
        )
        if not marker.is_file():
            return []
        return create_environment_hook(
            'ament_prefix_path', prefix_path, pkg_name,
            'AMENT_PREFIX_PATH', '', mode='prepend')

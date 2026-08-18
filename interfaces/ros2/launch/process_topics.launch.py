import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def resolve_config(value, subdir):
    """Resolve a config-file launch argument to a full path.

    An absolute path (starting with '/') is used verbatim so configs can live in
    an external folder; otherwise `value` is treated as a name (without .yaml)
    looked up in this package's installed config/<subdir> directory.
    """
    if value.startswith('/'):
        return value
    pkg_share = get_package_share_directory('bievr_lio_ros2')
    return os.path.join(pkg_share, 'config', subdir, value + '.yaml')


def launch_setup(context, *args, **kwargs):
    sensor_config = LaunchConfiguration('sensor_config').perform(context)
    params = LaunchConfiguration('params').perform(context)
    registration_metrics = LaunchConfiguration('registration_metrics').perform(context)
    use_sim_time_arg = LaunchConfiguration('use_sim_time').perform(context).lower()
    if use_sim_time_arg == 'auto':
        sensor_config_name = os.path.basename(sensor_config)
        use_sim_time = sensor_config_name in ('nav2_sim', 'nav2_sim.yaml')
    else:
        use_sim_time = use_sim_time_arg in ('true', '1', 'yes', 'on')

    rviz_config = os.path.join(
        get_package_share_directory('bievr_lio_ros2'), 'rviz', 'config.rviz')

    node_arguments = [
        '--sensor_config_file', resolve_config(sensor_config, 'sensor_configs'),
        '--params_file', resolve_config(params, ''),
    ]
    if registration_metrics:
        node_arguments.extend(['--registration_metrics', registration_metrics])

    return [
        Node(
            package='bievr_lio_ros2',
            executable='process_topics',
            name='bievr_lio_topics_node',
            output='screen',
            # Pass only the YAML config-file *paths* as command-line arguments;
            # the node parses them with yaml-cpp (the same plain files ROS1 uses).
            arguments=node_arguments,
            parameters=[{'use_sim_time': use_sim_time}],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            condition=IfCondition(LaunchConfiguration('rviz')),
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'sensor_config',
            description="Sensor config: a name (without .yaml) in config/sensor_configs/, "
                        "or an absolute path (starting with '/') to a config file."),
        DeclareLaunchArgument(
            'params', default_value='params',
            description="Algorithm params: a name (without .yaml) in config/, "
                        "or an absolute path (starting with '/') to a config file."),
        DeclareLaunchArgument(
            'rviz', default_value='false',
            description='Launch RViz2 with the bievr_lio visualization config.'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='auto',
            description='Use /clock; auto enables it for the nav2_sim sensor config.'),
        DeclareLaunchArgument(
            'registration_metrics', default_value='',
            description='Optional output path for the per-frame registration metrics CSV.'),
        OpaqueFunction(function=launch_setup),
    ])

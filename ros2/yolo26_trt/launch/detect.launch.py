from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("engine_path"),
            DeclareLaunchArgument("image_topic", default_value="/camera/color/image_raw"),
            DeclareLaunchArgument("output_topic", default_value="/yolo26/image"),
            DeclareLaunchArgument("conf_threshold", default_value="0.25"),
            Node(
                package="yolo26_trt",
                executable="yolo26_trt_node",
                name="yolo26_trt",
                output="screen",
                parameters=[
                    {
                        "engine_path": LaunchConfiguration("engine_path"),
                        "image_topic": LaunchConfiguration("image_topic"),
                        "output_topic": LaunchConfiguration("output_topic"),
                        "conf_threshold": LaunchConfiguration("conf_threshold"),
                    }
                ],
            ),
        ]
    )

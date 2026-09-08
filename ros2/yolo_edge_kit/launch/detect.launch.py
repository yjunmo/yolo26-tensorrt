from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("weights", default_value="yolov8n.pt"),
            DeclareLaunchArgument("image_topic", default_value="/camera/color/image_raw"),
            DeclareLaunchArgument("output_topic", default_value="/yolo/image"),
            DeclareLaunchArgument("imgsz", default_value="640"),
            DeclareLaunchArgument("conf", default_value="0.25"),
            DeclareLaunchArgument("device", default_value="0"),
            Node(
                package="yolo_edge_kit",
                executable="detect_node",
                name="yolo_detect",
                output="screen",
                parameters=[
                    {
                        "weights": LaunchConfiguration("weights"),
                        "image_topic": LaunchConfiguration("image_topic"),
                        "output_topic": LaunchConfiguration("output_topic"),
                        "imgsz": LaunchConfiguration("imgsz"),
                        "conf": LaunchConfiguration("conf"),
                        "device": LaunchConfiguration("device"),
                    }
                ],
            ),
        ]
    )

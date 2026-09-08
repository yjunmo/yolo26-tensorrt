from setuptools import setup
import os
from glob import glob

package_name = "yolo_edge_kit"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="seller",
    maintainer_email="seller@example.com",
    description="YOLO TensorRT ROS 2 detection node",
    license="MIT",
    entry_points={
        "console_scripts": [
            "detect_node = yolo_edge_kit.detect_node:main",
        ],
    },
)

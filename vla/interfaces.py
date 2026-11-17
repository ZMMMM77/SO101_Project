"""ROS topics and LeRobot feature names for the SO-101 data pipeline."""

CAMERAS = {
    "/so101/top/image_raw": "observation.images.top",
    "/so101/wrist/image_raw": "observation.images.wrist",
}
JOINT_STATES = "/so101/joint_states"
GRIPPER_POSITION = "/so101/gripper_position"
JOINT_TARGETS = "/so101/joint_targets"
GRIPPER_TARGET = "/so101/gripper_target"

TOPIC_TYPES = {
    **dict.fromkeys(CAMERAS, "sensor_msgs/msg/Image"),
    JOINT_STATES: "sensor_msgs/msg/JointState",
    GRIPPER_POSITION: "std_msgs/msg/Float64",
    JOINT_TARGETS: "sensor_msgs/msg/JointState",
    GRIPPER_TARGET: "std_msgs/msg/Float64",
}
CAMERA_RENAME_MAP = {
    "observation.images.top": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}

# Xbox 手柄控制 SO-101

## 当前默认控制架构

`run_safe_control.sh` 现已切换到 `xbox_official_control.py`。它保留此前可用的
`JOINT / XYZ` 两种控制模式，并加入 HOME 与 NEUTRAL 两个确定姿态；不再使用
逐关节归零展开。

- `folded_pose.json`：照片中的全折叠 `HOME/Ready` 启动与收纳位。
- `neutral_pose.json`：由用户手动摆放并采集的中立工作起始位。
- 从折叠位启动时显示 `LOCKED`，普通摇杆不会让机械臂运动。
- 按住 `RB` 并点按 `LB` 开始所有关节同步展开；全程保持 `RB`，松开立即取消并保持。
- 到达 `NEUTRAL` 后必须松开一次 `RB`，随后进入默认 `JOINT` 模式。
- 松开 RB 后按下左摇杆，可在 `JOINT / XYZ` 间切换。

适配对象：

- 微软 Xbox Wireless Controller
- Xbox Series X|S 标准版
- 常见背面型号：Model 1914
- Mac 通过蓝牙连接
- 复用项目根目录的 `so101_cartesian_control.py`
- 复用项目内 `.venv`

这套代码不会修改 LeRobot 的安装文件。Xbox 输入被转换成与 LeRobot F710
案例相同的动作结构：

```python
{
    "delta_x": float,
    "delta_y": float,
    "delta_z": float,
    "gripper": int,
}
```

## 控制逻辑

```text
Xbox SDL输入
  -> 10%摇杆死区
  -> LeRobot delta_x/delta_y/delta_z/gripper
  -> 读取SO-101关节状态
  -> FK计算当前末端位置
  -> 叠加小幅笛卡尔位移
  -> 工作空间和关节限位
  -> IK计算目标关节角
  -> LeRobot SO101Follower.send_action()
  -> Feetech Goal_Position
```

按键：

- 按住 `RB`：允许运动。松开立即保持当前姿态。
- 默认关节模式：左摇杆横向控制底座旋转，纵向控制肩关节。
- 默认关节模式：右摇杆纵向控制肘关节，横向控制腕部旋转。
- 默认关节模式：`Y` / `X` 分别控制腕部屈伸的两个方向。
- 松开 `RB` 后按下左摇杆：在关节模式和 XYZ 模式间切换。
- XYZ 模式：左摇杆控制末端 X/Y，右摇杆纵向控制 Z、横向控制腕部旋转。
- `RT`：打开夹爪。
- `LT`：关闭夹爪。
- 按住 `RB` 并点按 `LB`：在 HOME 与 NEUTRAL 之间平滑切换；松开 RB 立即取消。
- `B`：锁存软件急停，电机继续保持位置，不会突然卸力。
- 松开 `RB` 后按 `A`：解除软件急停。
- `VIEW`：显示当前关节角和 FK。
- `MENU`：退出；退出前要求用手托住机械臂，再释放扭矩。

普通游戏手柄不是工业安全急停。真机运行时必须有人守在电源开关旁。

## 旧实验版：折叠位置依次展开

以下内容只适用于保留下来的 `xbox_so101_control.py`，默认启动器已不再执行该流程。

程序启动后读取当前关节角，等待电机受力稳定，再重新捕获一次实测位置进行
保持。启动阶段不使用自定义安全区或标定限位判断。

如果关节2、3、4中任意一个距离 0° 超过 5°，程序进入 `UNFOLD`：

1. 按住 `RB`；
2. 将左摇杆向前推；
3. 关节4 `wrist_flex` 先向 0°缓慢移动；
4. 关节4到位后，关节3 `elbow_flex` 向 0°移动；
5. 最后关节2 `shoulder_lift` 向 0°移动。

一次只移动一个关节，反方向摇杆无效；松开摇杆或 `RB` 立即保持当前位置。
三个关节都进入 0°附近后，松开一次 `RB`，然后进入普通 XYZ/IK 控制。

安全启动器的展开速度为每秒 15°。该简化版本不做自定义启动限位或自动限位 E-stop，
但仍保留 `RB` 使能、手动 `B` 停止以及 LeRobot 原生单次指令限幅。

运行时状态行会持续显示五个关节的实测角度和当前目标角度：`P`、`S`、`E`、
`WF`、`WR` 分别对应底座、肩、肘、腕屈伸、腕旋转。

不连接手柄也可以自动执行一次小步诊断：

```bash
python -u xbox_so101_control.py auto-test --execute \
  --test-speed-deg-s 3 --test-motion-deg 15
```

它只选择展开顺序中第一个尚未到0°附近的关节，朝0°移动最多5°，逐周期打印
实测值和目标值；结束后保持扭矩并等待人工确认，不会自动释放机械臂。

## 一键返回原始位置

原始位置单独保存在 `original_pose_config.py`，不会与 FK/IK 或手柄映射混在
一起。机械臂尚未测量时，五个手臂关节保持为 `None`；此时按 `LB` 只会提示
尚未配置，不会产生运动。

连接机械臂并确认安全姿态后，把 VIEW/status 显示的 LeRobot 关节角填入：

```python
ORIGINAL_POSE_LEROBOT_DEG = {
    "shoulder_pan": None,
    "shoulder_lift": None,
    "elbow_flex": None,
    "wrist_flex": None,
    "wrist_roll": None,
    "gripper": None,
}
```

`gripper` 可以继续保留 `None`，表示回位时维持当前夹爪开度。自动回位过程中：

- 按 `B`：立即终止自动回位，锁存软件急停并保持当前位置。
- 按 `RB`：取消自动回位，切换回手动控制。
- 单次关节指令步长默认不超过 `0.75°`。

## 第一次运行顺序

激活已有环境：

```bash
cd /Users/luoziming/Desktop/SO101_Project
source ./activate_so101.zsh
cd xbox_controller
```

安装手柄读取依赖：

```bash
python -m pip install "pygame>=2.6,<3"
```

离线自检，不连接手柄和机械臂：

```bash
python xbox_so101_control.py self-test
python -m unittest -v
```

在 macOS“系统设置 -> 蓝牙”中连接 `Xbox Wireless Controller`，然后只测试输入：

```bash
python xbox_so101_control.py input-test
```

输入测试只在摇杆、扳机或按键的显示状态发生变化时输出一行；手柄静止时不会
持续重复刷屏。

依次确认：

1. 摇杆松开时所有轴接近 `0.00`。
2. 左摇杆能够改变 LeRobot 输出的 X/Y。
3. 右摇杆上下能够改变 Z，左右能够改变 `wr`。
4. RB、B、A、VIEW、MENU 显示正确。
5. LT、RT 能够分别显示夹爪关闭和打开。

只有输入全部正确后，才能运行真机：

```bash
python xbox_so101_control.py control --execute
```

首次真机联调优先从项目根目录运行保守参数启动器：

```bash
./xbox_controller/run_safe_control.sh
```

该启动器使用 10 Hz、5 mm/s、0.5 mm 单周期末端步长、8°单关节目标偏差、
15°/s 顺序展开速度、25°/s 手动关节速度，并在 300 秒后自动退出。macOS 蓝牙手柄输入需要从前台
Terminal 运行；后台或无图形会话可能无法收到 GameController 事件。

真机默认限制：

- 控制频率：15 Hz
- 末端速度：20 mm/s
- 单周期末端步长：最大 2 mm
- LeRobot 单次关节目标偏差：安全启动器最大 8°
- 启动展开速度：安全启动器 15°/s，可通过 `--unfold-speed-deg-s` 调整
- 不执行自定义启动限位或自动限位 E-stop
- IK规划被拒绝时保持当前位置，不锁存自动 E-stop
- 保留 LeRobot 单次关节目标变化限制和手动 `B` 停止

## 当前边界

- 支持直接控制五个手臂关节，也支持切换到末端 XYZ 加腕部旋转模式。
- SO-101只有五个手臂关节，任意位置加任意三轴姿态并不总是可解。
- 代码已按 Xbox 标准 SDL 映射编写，但手柄到货后仍必须运行 `input-test`
  确认 macOS、蓝牙固件和 SDL 报告一致。
- 本代码的“保持位置”是舵机位置闭环，不是主动重力补偿或柔顺控制。

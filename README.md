# SO-101 Control Demo

SO-101 机械臂在 MuJoCo 中将三个方块放入两个盒子。使用逆运动学、轨迹插值和反馈状态机，不使用学习模型或训练数据。

## 运行

Python 3.11 或 3.12。在本目录安装依赖后启动：

```bash
python -m pip install -r requirements.txt
python run_demo.py
```

首次运行自动下载固定版本的 SO-101 物理模型，需要访问 GitHub；以后使用本地缓存。也可以使用已有模型：

```bash
python run_demo.py --model "D:/so101/models/robotstudio_so101/scene.xml"
```

默认依次将红、黄方块放入绿色盒子，蓝方块放入紫色盒子。终端输出各阶段和最终结果。`--headless` 可不打开窗口运行。

## 外部指令

先启动等待指令的模式，再在另一个终端发送任务：

```bash
python run_demo.py --manual
```

```bash
python send_command.py pick red_cube left_box --id task-1
python send_command.py status
```

接口为本机 `http://127.0.0.1:8765`。`POST /commands` 接收以下 JSON，`GET /status` 查询执行结果。提交成功只表示已入队，任务完成以 `SUCCESS` 为准。

```json
{"id":"task-1","action":"pick_place","object_id":"red_cube","target_id":"left_box"}
```

物体为 `red_cube`、`blue_cube`、`yellow_cube`；目标为 `left_box`（绿色，机器人 +Y 侧）或 `right_box`（紫色，−Y 侧）。

## 代码

- `control.py`：场景、逆运动学、抓放状态机和任务调度。检查到位、夹持、抬升、释放和落位反馈，再进入下一阶段。
- `run_demo.py`：模型下载、本机 JSON 接口及仿真循环。接口接收任务，物理线程更新控制输入；执行结果通过同一任务 ID 返回。
- `send_command.py`：发送任务和查询状态的命令行客户端。

采用“指令—持续执行—状态反馈”的控制流程。

## 当前范围

原模块版已在 Windows 上完成三物体分拣及外部指令抓放；整理版在 MuJoCo 中复核。物体坐标和接触信息直接来自仿真。当前支持指定布局中的直立方块，不包含视觉定位、通用避障或真机适配。

物理模型来自 [MuJoCo Menagerie / robotstudio_so101](https://github.com/google-deepmind/mujoco_menagerie/tree/98a70c569c4c1b6da467a20012f766ee1f8dc521/robotstudio_so101)，下载时一并保留上游 LICENSE。运行时增加方块和盒子，并调整夹爪力矩限制。

## 真机复现

当前版本是仿真示例，尚未接入或验证 SO-101 真机。固定的是抓放流程；每次任务开始时会读取物体位置，再用逆运动学计算关节目标，并非回放一组固定关节角。

不要求使用完全相同的方块，但首次复现建议使用夹爪能够夹住的直立硬质方块，沿用从上方抓取的方式。更换尺寸后，需要调整抓取高度、夹爪开口、搬运高度和放置间距；杯子、碗或软物体需要另外设计抓点和姿态。

真机运行前还需完成：

- **硬件与校准**：接入电机通信，校准关节零位、方向、单位和限位，将关节目标映射到实际电机，并设置合适的速度与夹持力度。
- **坐标与尺寸**：用人工测量或相机定位提供物体和盒子的位置，统一到机械臂基座坐标系；按实际尺寸调整参数。当前代码直接读取 MuJoCo 状态，需要替换这部分读取。
- **执行反馈**：用真机可获得的关节、夹爪反馈及必要的视觉确认，替换仿真的接触、抬升和落位判断。夹爪闭合不等于成功抓住物体。

首次验证可固定机械臂、方块起点和盒子位置，在桌面做标记，测量坐标后低速完成一次抓放；跑通后再加入相机定位。固定摆位的验证无需学习模型或训练数据，但上述适配尚未包含在本仓库中。

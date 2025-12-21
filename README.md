# Highway Incident Detection System

工业级高速公路事件监测系统，基于YOLO视觉算法实现目标检测和多种事件识别。

## 功能特性

- **多事件检测**：
  - 道路施工（施工车、工人）
  - 车辆逆行
  - 摩托车
  - 行人
  - 路面抛洒物

- **多路RTSP流处理**：支持同时处理多个RTSP视频流，高效异步处理
- **自动保活机制**：视频流自动重连、推理进程自动重启
- **资源管理**：完善的资源监控、内存管理和故障恢复
- **实时推理**：基于YOLO模型的高效目标检测
- **异步摘要生成**：不阻塞实时推理的摘要、关键帧和视频生成
- **可配置参数**：YOLO模型参数、事件参数、视频流参数均可配置
- **高效日志系统**：精准的日志记录，便于复现与跟踪分析

## 项目结构

```
HighwayIncidentDetection/
├── highway_incident_detection/    # 主包
│   ├── __init__.py
│   ├── config/                    # 配置管理
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── models/                    # YOLO模型封装
│   │   ├── __init__.py
│   │   └── yolo_model.py
│   ├── detectors/                 # 事件检测器
│   │   ├── __init__.py
│   │   ├── event_types.py
│   │   └── event_detector.py
│   ├── stream/                    # RTSP流处理
│   │   ├── __init__.py
│   │   └── rtsp_stream.py
│   ├── summary/                   # 摘要生成
│   │   ├── __init__.py
│   │   └── summary_generator.py
│   └── utils/                     # 工具函数
│       ├── __init__.py
│       ├── logger.py
│       └── tracker.py
├── config/                        # 配置文件目录
│   └── config.yaml
├── logs/                          # 日志目录
├── output/                        # 输出目录
│   ├── keyframes/                 # 关键帧图片
│   ├── videos/                    # 事件视频片段
│   └── summaries/                 # 摘要JSON文件
├── main.py                        # 主程序入口（向后兼容）
├── pyproject.toml                 # 现代打包配置（PEP 518/621）
├── requirements.txt               # 依赖包（传统方式）
├── MANIFEST.in                    # 打包清单
├── README.md                      # 本文档
└── INSTALLATION.md                # 详细安装指南
```

## 安装

### 方式一：使用 pip 安装（推荐）

```bash
# 从源码安装
pip install .

# 或者以开发模式安装（可编辑模式）
pip install -e .

# 安装开发依赖
pip install -e ".[dev]"
```

### 方式二：使用 requirements.txt（传统方式）

```bash
pip install -r requirements.txt
```

### 方式三：从 Git 仓库安装

```bash
pip install git+https://github.com/M4D-AI/highway-incident-detection.git
```

### 配置

编辑 `config/config.yaml` 文件，配置以下参数：

- **YOLO模型参数**：模型路径、置信度阈值、IOU阈值、设备等
- **事件参数**：各类事件的检测阈值、冷却时间等
- **RTSP流参数**：流地址、帧跳过、缓冲区大小等
- **输出参数**：输出目录、视频时长、FPS等
- **日志参数**：日志目录、级别、轮转等

### 准备YOLO模型

系统使用Ultralytics YOLO模型。首次运行时会自动下载默认模型（yolov8n.pt），也可以指定自定义模型路径。

## 使用方法

### 方式一：使用命令行工具（推荐）

安装后可以直接使用命令行工具：

```bash
# 使用默认配置
highway-incident-detection

# 使用自定义配置文件
highway-incident-detection --config config/config.yaml

# 查看版本
highway-incident-detection --version
```

### 方式二：使用 Python 脚本

```bash
# 基本使用
python main.py

# 使用自定义配置文件
python main.py --config config/config.yaml
```

### 方式三：作为 Python 模块使用

```python
from highway_incident_detection import HighwayIncidentDetectionSystem
import asyncio

system = HighwayIncidentDetectionSystem(config_path="config/config.yaml")
asyncio.run(system.run())
```

## 配置说明

### YOLO配置

```yaml
yolo:
  model_path: "yolov8n.pt"          # YOLO模型路径
  confidence_threshold: 0.5         # 置信度阈值
  iou_threshold: 0.45               # IOU阈值
  device: "cuda"                    # 设备 (cuda/cpu)
  img_size: 640                     # 输入图像尺寸
```

### 事件配置

```yaml
event:
  construction_vehicle_class_ids: [2, 3, 5, 7]  # 施工车辆类别ID
  worker_class_id: 0                             # 工人类别ID
  motorcycle_class_id: 3                         # 摩托车类别ID
  pedestrian_class_id: 0                          # 行人类别ID
  debris_class_ids: [24, 25, 26, 27, 28]        # 抛洒物类别ID
  construction_detection_frames: 10              # 施工检测帧数
  reverse_detection_frames: 15                   # 逆行检测帧数
  reverse_speed_threshold: -0.5                  # 逆行速度阈值
  event_cooldown_seconds: 30                      # 事件冷却时间（秒）
```

### RTSP流配置

```yaml
stream:
  streams:
    - name: "stream_1"
      url: "rtsp://example.com/stream1"
      enabled: true
    - name: "stream_2"
      url: "rtsp://example.com/stream2"
      enabled: false
  frame_skip: 1                    # 帧跳过（处理每N帧）
  buffer_size: 30                  # 缓冲区大小
  reconnect_interval: 5            # 重连间隔（秒）
  max_reconnect_attempts: 10       # 最大重连次数
  read_timeout: 10                 # 读取超时（秒）
```

### 输出配置

```yaml
output:
  output_dir: "./output"
  keyframe_dir: "./output/keyframes"
  video_dir: "./output/videos"
  summary_dir: "./output/summaries"
  video_duration_seconds: 10       # 视频时长（秒）
  video_fps: 25                    # 视频FPS
```

## 输出说明

系统会为每个检测到的事件生成：

1. **关键帧图片**：包含事件标注的图片，保存在 `output/keyframes/`
2. **视频片段**：10秒的事件视频，保存在 `output/videos/`
3. **摘要JSON**：包含事件详细信息的JSON文件，保存在 `output/summaries/`

摘要JSON格式：

```json
{
  "event_id": "stream_1_construction_2024-01-01T12:00:00",
  "event_type": "construction",
  "stream_name": "stream_1",
  "timestamp": "2024-01-01T12:00:00",
  "frame_number": 12345,
  "confidence": 0.85,
  "description": "Construction detected: 2 vehicles, 3 workers",
  "bbox": [100, 200, 300, 400],
  "metadata": {
    "vehicle_count": 2,
    "worker_count": 3
  },
  "keyframe_path": "./output/keyframes/...",
  "video_path": "./output/videos/..."
}
```

## 日志

日志文件保存在 `logs/` 目录：

- `highway_incident_YYYY-MM-DD.log`：所有日志
- `errors_YYYY-MM-DD.log`：错误日志

日志自动轮转和压缩，便于长期保存和分析。

## 资源管理与自动保活

系统实现了完善的资源管理和自动保活机制：

- **视频流自动重连**：无限重连机制，自动检测连接状态，失败时自动重连
- **心跳检测**：定期检查流连接健康状态，无帧超时自动触发重连
- **推理进程自动重启**：模型故障时自动重启，确保服务连续性
- **资源监控**：实时监控内存、CPU使用情况，自动触发垃圾回收
- **故障恢复**：完善的异常处理和资源清理机制

## 性能优化

- **多路流处理**：使用异步处理，支持同时处理多个RTSP流
- **帧跳过**：可配置帧跳过，降低计算负载
- **异步摘要生成**：摘要生成不阻塞实时推理
- **高效推理**：支持GPU加速，批量推理优化
- **资源管理**：自动资源清理和内存管理

## 代码风格

代码遵循Google Python代码风格规范。

## 许可证

[Apache License 2.0]

## 贡献

欢迎提交Issue和Pull Request。

## 联系方式

[xiezl_cjlu@foxmail.com]


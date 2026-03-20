# Highway Incident Detection System

高速公路事件监测系统，基于 YOLO 做目标检测，支持多路视频流、事件识别、证据落盘、结果外发，以及大模型二次审计。

## 适合做什么

- 高速公路 / 园区道路视频事件检测
- RTSP 或本地视频流接入
- 事件结果外发到 Webhook、Kafka、Redis 或自定义系统
- 通过自定义 detector 快速扩展业务规则

## 核心能力

- 内置事件：施工、逆行、摩托车、行人、抛洒物、异常停车、拥堵
- 多路流处理与背压控制
- 关键帧、视频片段、摘要 JSON 输出
- 多模态大模型审计与风险分级
- 自定义结果 handler
- 自定义事件 detector

## 快速开始

### 安装

```bash
pip install .
```

开发环境：

```bash
pip install -e ".[dev]"
```

如果要使用 Kafka / Redis handler：

```bash
pip install -e ".[integrations]"
```

### 运行

```bash
highway-incident-detection --config config/config.yaml
```

或：

```bash
python main.py --config config/config.yaml
```

## 最常用配置

默认配置在 [config/config.yaml](/Users/xiezhanlong/PycharmProjects/HighwayIncidentDetection/config/config.yaml)。

```yaml
yolo:
  model_path: "yolov8n.pt"
  confidence_threshold: 0.5
  device: "cpu"

stream:
  frame_skip: 1
  max_pending_frames: 2
  drop_frames_when_busy: true

audit:
  enabled: false
  model_name: "gpt-4.1-mini"
```

## 输出结果

系统默认会生成：

- `output/keyframes/`
- `output/videos/`
- `output/summaries/`
- `output/audits/`

同时支持对外推送统一结果对象 `EventResult`。

## 结果接口

### 自定义 handler

```python
from highway_incident_detection import EventResult, HighwayIncidentDetectionSystem


async def handle_result(result: EventResult) -> None:
    print(result.to_dict())


system = HighwayIncidentDetectionSystem(config_path="config/config.yaml")
system.register_result_handler(handle_result)
```

### 内置 handler

```python
from highway_incident_detection import (
    HTTPWebhookHandler,
    KafkaHandler,
    RedisHandler,
)

system.register_result_handler(
    HTTPWebhookHandler("http://127.0.0.1:8080/webhook")
)
system.register_result_handler(
    KafkaHandler(topic="incident-events", bootstrap_servers="127.0.0.1:9092")
)
system.register_result_handler(
    RedisHandler(url="redis://127.0.0.1:6379/0", channel="incident-events")
)
```

## 自定义事件扩展

### 代码注册

```python
from highway_incident_detection import EventDetectionContext, HighwayIncidentDetectionSystem


def smoke_alert_detector(detector, ctx: EventDetectionContext):
    if detector.is_in_cooldown(ctx.stream_name, "smoke_alert", ctx.timestamp):
        return None

    detector.update_event_time(ctx.stream_name, "smoke_alert", ctx.timestamp)
    return detector.create_event(
        event_type="smoke_alert",
        stream_name=ctx.stream_name,
        timestamp=ctx.timestamp,
        frame_number=ctx.frame_number,
        confidence=0.9,
        description="Custom smoke alert",
    )


system = HighwayIncidentDetectionSystem(config_path="config/config.yaml")
system.register_event_detector(
    name="smoke_alert_rule",
    event_type="smoke_alert",
    handler=smoke_alert_detector,
)
```

### 配置自动加载

示例模块在 [custom_detectors/smoke.py](/Users/xiezhanlong/PycharmProjects/HighwayIncidentDetection/custom_detectors/smoke.py)。

```yaml
event:
  custom_detectors:
    - name: "smoke_alert_rule"
      event_type: "smoke_alert"
      handler: "custom_detectors.smoke:smoke_alert_detector"
      enabled: true
      prepend: true
```

## Benchmark

```bash
python benchmarks/pipeline_benchmark.py \
  --video data/parking.mp4 \
  --duration-seconds 10 \
  --frame-skips 1,2,4 \
  --max-pending-frames 1,2,4 \
  --processing-ms 20 \
  --handler-profiles none,sync,async,mixed \
  --handler-delay-ms 10 \
  --emit-every-n-frames 30
```

## 文档

- 使用者文档：当前文件
- 开发者文档：[DEVELOPER.md](/Users/xiezhanlong/PycharmProjects/HighwayIncidentDetection/DEVELOPER.md)
- 安装细节：[INSTALLATION.md](/Users/xiezhanlong/PycharmProjects/HighwayIncidentDetection/INSTALLATION.md)

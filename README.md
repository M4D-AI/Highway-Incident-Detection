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

GPU 多流场景建议直接从 [config/gpu_batch_config.yaml](/Users/xiezhanlong/PycharmProjects/HighwayIncidentDetection/config/gpu_batch_config.yaml) 起步：

```bash
highway-incident-detection --config config/gpu_batch_config.yaml
```

## 最常用配置

默认配置在 [config/config.yaml](/Users/xiezhanlong/PycharmProjects/HighwayIncidentDetection/config/config.yaml)。
GPU 微批示例在 [config/gpu_batch_config.yaml](/Users/xiezhanlong/PycharmProjects/HighwayIncidentDetection/config/gpu_batch_config.yaml)。

```yaml
yolo:
  model_path: "yolov8n.pt"
  confidence_threshold: 0.5
  device: "cpu"
  batch_inference_enabled: true
  max_batch_size: 4
  batch_timeout_ms: 15

stream:
  frame_skip: 1
  max_pending_frames: 2
  drop_frames_when_busy: true

audit:
  enabled: false
  model_name: "gpt-4.1-mini"
  user_prompt_path: "./prompts/audit_user.txt"
```

## 输出结果

系统默认会生成：

- `output/keyframes/`
- `output/videos/`
- `output/summaries/`
- `output/audits/`

同时支持对外推送统一结果对象 `EventResult`。
`summary JSON` 和 `EventResult` 会带 `processing_metrics`，可直接拿到批大小、排队等待和后处理等待。

## 审计 Prompt 自定义

审计模块支持直接改 prompt，不需要改代码。可选两种方式：

- 在配置里直接写 `audit.system_prompt` 和 `audit.user_prompt_template`
- 用 `audit.system_prompt_path` 和 `audit.user_prompt_path` 指向外部文本文件
- 用 `audit.event_prompt_overrides.<event_type>` 为某个事件单独覆盖 prompt，未命中时仍走全局默认

`user_prompt_template` 支持这些占位符：

- `{event_type}`
- `{confidence}`
- `{description}`
- `{stream_name}`
- `{frame_number}`
- `{timestamp}`
- `{metadata_json}`
- `{risk_levels}`

示例：

```yaml
audit:
  enabled: true
  endpoint: "https://api.openai.com/v1/chat/completions"
  api_key: "${OPENAI_API_KEY}"
  system_prompt: "你是高速事件复核员，只返回 JSON。"
  user_prompt_template: |
    请复核以下事件是否真实：
    事件类型: {event_type}
    置信度: {confidence:.2f}
    流名称: {stream_name}
    帧号: {frame_number}
    描述: {description}
    元数据: {metadata_json}
```

按事件类型覆盖的示例：

```yaml
audit:
  system_prompt: "你是高速事件复核员，只返回 JSON。"
  user_prompt_template: "请复核事件 {event_type}。"
  event_prompt_overrides:
    pedestrian:
      system_prompt: "你专门复核行人闯入事件，只返回 JSON。"
      user_prompt_template: |
        请重点判断是否真有行人进入车道：
        流名称: {stream_name}
        帧号: {frame_number}
        描述: {description}
```

## 结果接口

### 自定义 handler

```python
from highway_incident_detection import EventResult, HighwayIncidentDetectionSystem


async def handle_result(result: EventResult) -> None:
    payload = result.to_dict()
    print(payload["event_type"], payload["processing_metrics"])


system = HighwayIncidentDetectionSystem(config_path="config/config.yaml")
system.register_result_handler(handle_result)
```

运行中也可以直接读系统快照：

```python
stats = system.get_runtime_stats()
print(stats["batch_inference"]["avg_batch_size"])
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

如果是 GPU 多流场景，建议重点调这几个参数：

- `yolo.batch_inference_enabled`
- `yolo.max_batch_size`
- `yolo.batch_timeout_ms`
- `yolo.batch_queue_size`
- `stream.max_pending_frames`

重点观察这些运行时指标：

- `avg_batch_size`
- `timeout_batches`
- `avg_queue_wait_ms`
- `avg_postprocess_wait_ms`

## 文档

- 使用者文档：当前文件
- 开发者文档：[DEVELOPER.md](/Users/xiezhanlong/PycharmProjects/HighwayIncidentDetection/DEVELOPER.md)
- 安装细节：[INSTALLATION.md](/Users/xiezhanlong/PycharmProjects/HighwayIncidentDetection/INSTALLATION.md)

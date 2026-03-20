# Developer Guide

面向二次开发、扩展和性能调优。

## 系统架构

```mermaid
flowchart LR
    A[RTSP / Video] --> B[RTSPStreamProcessor]
    B --> C[Bounded Scheduling / Backpressure]
    C --> D[YOLOModel]
    D --> E[ROI Filter + Tracker]
    E --> F[EventDetector]
    F --> G[Event Queue]
    G --> H[SummaryGenerator]
    H --> I[Keyframe / Video / Summary]
    H --> J[LLM Audit]
    H --> K[Result Handlers]
    K --> L[Custom Handler]
    K --> M[HTTPWebhookHandler]
    K --> N[KafkaHandler]
    K --> O[RedisHandler]
```

## 模块职责

- `highway_incident_detection/system.py`
  系统装配、生命周期管理、组件初始化、事件队列、微批推理调度，以及批处理运行时统计。

- `highway_incident_detection/stream/rtsp_stream.py`
  取流、重连、心跳、背压控制、在途帧上限控制。

- `highway_incident_detection/models/yolo_model.py`
  YOLO 推理封装、单帧/批量推理、失败恢复、推理统计。

- `highway_incident_detection/detectors/event_detector.py`
  内置事件检测逻辑与注册式 detector 扩展入口。

- `highway_incident_detection/detectors/loader.py`
  从 `module:callable` 配置自动加载自定义 detector。

- `highway_incident_detection/summary/summary_generator.py`
  关键帧、事件视频、summary、audit record 生成，以及结果派发。

- `highway_incident_detection/output/`
  `EventResult`、内置 handler、对外结果输出协议。

- `highway_incident_detection/audit/event_auditor.py`
  大模型二次审计与本地风险回退策略。

## 自定义事件扩展

### 接口约定

自定义 detector 统一签名：

```python
def my_detector(detector, ctx: EventDetectionContext):
    ...
```

可用能力：

- `detector.is_in_cooldown(...)`
- `detector.update_event_time(...)`
- `detector.create_event(...)`

返回值可以是：

- `None`
- 单个 `IncidentEvent`
- `IncidentEvent` 列表

### 代码注册

```python
system.register_event_detector(
    name="smoke_alert_rule",
    event_type="smoke_alert",
    handler=my_detector,
    prepend=True,
)
```

### 配置自动加载

```yaml
event:
  custom_detectors:
    - name: "smoke_alert_rule"
      event_type: "smoke_alert"
      handler: "custom_detectors.smoke:smoke_alert_detector"
      enabled: true
      prepend: true
```

示例模块：
[custom_detectors/smoke.py](/Users/xiezhanlong/PycharmProjects/HighwayIncidentDetection/custom_detectors/smoke.py)

## 结果输出扩展

统一结果对象：

- `EventResult`

新增字段：

- `processing_metrics`

注册方式：

```python
system.register_result_handler(custom_handler)
```

内置 handler：

- `HTTPWebhookHandler`
- `KafkaHandler`
- `RedisHandler`

## 审计 Prompt 扩展

审计模块默认内置 system prompt 和 user prompt template，也支持通过配置覆盖：

- `audit.system_prompt`
- `audit.user_prompt_template`
- `audit.system_prompt_path`
- `audit.user_prompt_path`
- `audit.event_prompt_overrides.<event_type>`

`user_prompt_template` 当前可用占位符：

- `{event_type}`
- `{confidence}`
- `{description}`
- `{stream_name}`
- `{frame_number}`
- `{timestamp}`
- `{metadata_json}`
- `{risk_levels}`

覆盖优先级：

- 事件级 `prompt_path`
- 事件级内联 prompt
- 全局 `prompt_path`
- 全局内联 prompt
- 内置默认 prompt

## 性能调优点

重点参数在 [config/config.yaml](/Users/xiezhanlong/PycharmProjects/HighwayIncidentDetection/config/config.yaml)：

- `stream.frame_skip`
  降低推理频率，直接影响吞吐。

- `stream.max_pending_frames`
  控制每路流最大在途帧数，值越大越能吸收短时抖动，但会增加积压风险。

- `stream.drop_frames_when_busy`
  在处理跟不上时是否直接丢帧。

- `yolo.img_size`
  影响精度与推理速度。

- `yolo.batch_inference_enabled`
  GPU 多流场景下启用微批检测。

- `yolo.max_batch_size`
  决定一次推理最多合并多少帧。

- `yolo.batch_timeout_ms`
  决定为了凑批最多等待多久；越大吞吐越高，但延迟也可能更高。

- `yolo.batch_queue_size`
  决定待推理帧队列容量。

- `system.get_runtime_stats()`
  直接获取模型、流状态和微批运行时统计快照。

- `audit.enabled`
  审计会增加事件后处理时延。

## Benchmark

运行：

```bash
python benchmarks/pipeline_benchmark.py \
  --video data/parking.mp4 \
  --duration-seconds 10 \
  --frame-skips 1,2,4 \
  --max-pending-frames 1,2,4 \
  --processing-ms 20 \
  --handler-profiles none,sync,async,mixed \
  --handler-delay-ms 10 \
  --emit-every-n-frames 30 \
  --output-json output/benchmark/report.json
```

关注指标：

- `processed_frames`
- `dropped_frames`
- `effective_fps`
- `avg_callback_ms / p95_callback_ms`
- `avg_emit_ms / p95_emit_ms`

如果是在线 GPU 多流运行，还建议同时观察：

- `avg_batch_size`
- `timeout_batches`
- `avg_queue_wait_ms`
- `avg_postprocess_wait_ms`

## 开发验证

```bash
pytest -q -o addopts=''
python -m compileall highway_incident_detection custom_detectors benchmarks tests example_usage.py main.py
```

## 关键目录

```text
highway_incident_detection/
custom_detectors/
benchmarks/
tests/
```

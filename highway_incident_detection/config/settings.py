"""Configuration settings using Pydantic."""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class YOLOConfig(BaseSettings):
    """YOLO model configuration."""

    model_path: str = Field(
        default="yolov8n.pt",
        description="Path to YOLO model file"
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for detections"
    )
    iou_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="IOU threshold for NMS"
    )
    device: str = Field(
        default="cuda",
        description="Device to run inference on (cuda/cpu)"
    )
    img_size: int = Field(
        default=640,
        ge=320,
        le=1280,
        description="Input image size for inference"
    )
    auto_restart: bool = Field(
        default=True,
        description="Enable automatic model restart on failure"
    )
    health_check_interval: int = Field(
        default=300,
        ge=1,
        description="Model health check interval in seconds"
    )
    max_failures: int = Field(
        default=5,
        ge=1,
        description="Maximum consecutive failures before restart"
    )
    batch_inference_enabled: bool = Field(
        default=False,
        description="Enable micro-batch inference across multiple streams"
    )
    max_batch_size: int = Field(
        default=4,
        ge=1,
        le=64,
        description="Maximum number of frames in one inference batch"
    )
    batch_timeout_ms: int = Field(
        default=15,
        ge=1,
        le=1000,
        description="Maximum wait time in milliseconds to form an inference batch"
    )
    batch_queue_size: int = Field(
        default=32,
        ge=1,
        description="Maximum queued frames waiting for batch inference"
    )

    model_config = SettingsConfigDict(extra="forbid")


class EventConfig(BaseSettings):
    """Event detection configuration."""

    # Class IDs for different events.
    vehicle_class_ids: List[int] = Field(
        default=[2, 3, 5, 7],
        description="Class IDs treated as vehicles for reverse, parking, and congestion detection"
    )
    construction_vehicle_class_ids: List[int] = Field(
        default=[2, 3, 5, 7],
        description="Class IDs that indicate a construction scene"
    )
    worker_class_id: int = Field(
        default=0,
        description="Class ID for workers"
    )
    construction_require_worker: bool = Field(
        default=True,
        description="Whether construction detection requires worker class to appear together with construction classes"
    )
    motorcycle_class_id: int = Field(
        default=3,
        description="Class ID for motorcycles"
    )
    pedestrian_class_id: int = Field(
        default=0,
        description="Class ID for pedestrians"
    )
    debris_class_ids: List[int] = Field(
        default=[24, 25, 26, 27, 28],
        description="Class IDs for road debris"
    )

    # Detection thresholds
    construction_detection_frames: int = Field(
        default=10,
        ge=1,
        description="Number of consecutive frames to confirm construction"
    )
    reverse_detection_frames: int = Field(
        default=15,
        ge=1,
        description="Number of frames to track for reverse detection"
    )
    reverse_speed_threshold: float = Field(
        default=-0.5,
        description="Speed threshold (pixels/frame) for reverse detection"
    )
    event_cooldown_seconds: int = Field(
        default=30,
        ge=0,
        description="Cooldown period between same event detections (seconds)"
    )
    # Abnormal parking detection
    parking_detection_frames: int = Field(
        default=60,
        ge=1,
        description="Number of frames a vehicle must be stationary to trigger parking event"
    )
    parking_speed_threshold: float = Field(
        default=0.5,
        ge=0.0,
        description="Maximum speed (pixels/frame) to consider vehicle as stationary"
    )
    parking_position_threshold: float = Field(
        default=10.0,
        ge=0.0,
        description="Maximum position change (pixels) to consider vehicle as stationary"
    )
    # Congestion detection
    congestion_min_vehicles: int = Field(
        default=5,
        ge=1,
        description="Minimum number of vehicles to trigger congestion event"
    )
    congestion_speed_threshold: float = Field(
        default=2.0,
        ge=0.0,
        description="Maximum average speed (pixels/frame) to consider as congestion"
    )
    congestion_detection_frames: int = Field(
        default=30,
        ge=1,
        description="Number of consecutive frames to confirm congestion"
    )
    congestion_density_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum vehicle density (vehicles per unit area) to trigger congestion"
    )
    custom_detectors: List["CustomEventDetectorConfig"] = Field(
        default_factory=list,
        description="Custom detector registrations loaded from module:callable paths"
    )

    model_config = SettingsConfigDict(extra="forbid")


class CustomEventDetectorConfig(BaseSettings):
    """Custom detector registration config."""

    name: str = Field(
        description="Unique detector name"
    )
    event_type: str = Field(
        description="Event type string emitted by the custom detector"
    )
    handler: str = Field(
        description="Callable path in module:callable format"
    )
    enabled: bool = Field(
        default=True,
        description="Whether this custom detector should be loaded"
    )
    prepend: bool = Field(
        default=False,
        description="Whether to run this detector before built-in detectors"
    )

    model_config = SettingsConfigDict(extra="forbid")


class StreamConfig(BaseSettings):
    """RTSP stream configuration."""

    streams: List[Dict[str, Any]] = Field(
        default_factory=lambda: [
            {
                "name": "stream_1",
                "url": "rtsp://example.com/stream1",
                "enabled": True,
            }
        ],
        description="List of RTSP stream configurations"
    )
    frame_skip: int = Field(
        default=1,
        ge=1,
        description="Process every Nth frame (1 = all frames)"
    )
    buffer_size: int = Field(
        default=30,
        ge=1,
        description="Frame buffer size for each stream"
    )
    reconnect_interval: int = Field(
        default=5,
        ge=1,
        description="Reconnection interval in seconds"
    )
    max_reconnect_attempts: int = Field(
        default=-1,
        description="Maximum reconnection attempts (-1 for infinite)"
    )
    read_timeout: int = Field(
        default=10,
        ge=1,
        description="Stream read timeout in seconds"
    )
    heartbeat_interval: int = Field(
        default=30,
        ge=1,
        description="Heartbeat check interval in seconds"
    )
    no_frame_timeout: int = Field(
        default=60,
        ge=1,
        description="Timeout for no frames received (seconds)"
    )
    auto_reconnect: bool = Field(
        default=True,
        description="Enable automatic reconnection"
    )
    max_pending_frames: int = Field(
        default=2,
        ge=1,
        description="Maximum in-flight frame processing tasks per stream"
    )
    drop_frames_when_busy: bool = Field(
        default=True,
        description="Drop new frames when in-flight processing reaches the limit"
    )

    model_config = SettingsConfigDict(extra="forbid")


class LaneZoneConfig(BaseSettings):
    """One lane zone with its own ROI and expected direction."""

    name: str = Field(
        default="default",
        description="Unique lane zone name within one stream"
    )
    roi_points: List[Tuple[int, int]] = Field(
        default_factory=list,
        description="ROI polygon points for this lane zone"
    )
    direction: Optional[str] = Field(
        default=None,
        description="Expected direction for this lane zone: 'left', 'right', 'up', 'down', or None"
    )

    model_config = SettingsConfigDict(extra="forbid")


class LaneConfig(BaseSettings):
    """Lane configuration for ROI and direction."""

    enabled: bool = Field(
        default=False,
        description="Enable lane-based detection (only detect events in lane ROI)"
    )
    # ROI region: list of points defining polygon [(x1, y1), (x2, y2), ...]
    # Empty list means use full frame
    roi_points: List[Tuple[int, int]] = Field(
        default_factory=list,
        description="ROI polygon points for lane area. Empty list = full frame"
    )
    # Lane direction: "left", "right", "up", "down", or None for bidirectional
    direction: Optional[str] = Field(
        default=None,
        description="Lane direction: 'left', 'right', 'up', 'down', or None for bidirectional"
    )
    zones: List[LaneZoneConfig] = Field(
        default_factory=list,
        description="Optional multiple lane zones, each with its own ROI and direction"
    )
    roi_reference_size: Optional[Tuple[int, int]] = Field(
        default=None,
        description=(
            "[width, height] of the image roi_points were drawn on. ROI points are "
            "scaled to the actual frame size at runtime. None = use points as-is"
        )
    )

    model_config = SettingsConfigDict(extra="forbid")


class OutputConfig(BaseSettings):
    """Output configuration."""

    output_dir: str = Field(
        default="./output",
        description="Output directory for results"
    )
    keyframe_dir: str = Field(
        default="./output/keyframes",
        description="Directory for keyframe images"
    )
    video_dir: str = Field(
        default="./output/videos",
        description="Directory for event videos"
    )
    summary_dir: str = Field(
        default="./output/summaries",
        description="Directory for summary files"
    )
    audit_dir: str = Field(
        default="./output/audits",
        description="Directory for audit files"
    )
    video_duration_seconds: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Duration of event video clips in seconds"
    )
    video_fps: int = Field(
        default=25,
        ge=10,
        le=60,
        description="FPS for output videos"
    )

    model_config = SettingsConfigDict(extra="forbid")


class AuditPromptOverrideConfig(BaseSettings):
    """Per-event audit prompt override configuration."""

    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt override for a specific event type"
    )
    user_prompt_template: Optional[str] = Field(
        default=None,
        description="Optional user prompt template override for a specific event type"
    )
    system_prompt_path: Optional[str] = Field(
        default=None,
        description="Optional system prompt file override for a specific event type"
    )
    user_prompt_path: Optional[str] = Field(
        default=None,
        description="Optional user prompt file override for a specific event type"
    )

    model_config = SettingsConfigDict(extra="forbid")


class AuditConfig(BaseSettings):
    """Large-model audit configuration."""

    enabled: bool = Field(
        default=False,
        description="Enable large-model audit for detected events"
    )
    provider: str = Field(
        default="openai_compatible",
        description="Audit provider type"
    )
    endpoint: Optional[str] = Field(
        default=None,
        description="OpenAI-compatible chat completion endpoint"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key used by the audit provider"
    )
    model_name: str = Field(
        default="gpt-4.1-mini",
        description="Model name used for event audit"
    )
    system_prompt: str = Field(
        default=(
            "You are reviewing highway incident detections. "
            "Return strict JSON with keys: confirmed, risk_level, confidence, "
            "reason, recommended_action."
        ),
        description="System prompt sent to the audit model"
    )
    user_prompt_template: str = Field(
        default=(
            "Review the detector result for a highway-monitoring system and decide "
            "whether the event is real. "
            "Event type: {event_type}. "
            "Detector confidence: {confidence:.2f}. "
            "Description: {description}. "
            "Stream name: {stream_name}. "
            "Frame number: {frame_number}. "
            "Timestamp: {timestamp}. "
            "Metadata: {metadata_json}. "
            "Classify risk_level as one of: {risk_levels}. "
            "If the evidence is weak, set confirmed to false."
        ),
        description="User prompt template sent to the audit model"
    )
    system_prompt_path: Optional[str] = Field(
        default=None,
        description="Optional file path for the system prompt template"
    )
    user_prompt_path: Optional[str] = Field(
        default=None,
        description="Optional file path for the user prompt template"
    )
    event_prompt_overrides: Dict[str, AuditPromptOverrideConfig] = Field(
        default_factory=dict,
        description="Per-event prompt overrides keyed by event type string"
    )
    timeout_seconds: int = Field(
        default=20,
        ge=1,
        le=120,
        description="Timeout in seconds for a single audit request"
    )
    max_images: int = Field(
        default=3,
        ge=1,
        le=6,
        description="Maximum number of images sampled for each audit"
    )
    sample_video_frames: bool = Field(
        default=True,
        description="Whether to sample frames from the generated event video"
    )
    min_event_confidence: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum detector confidence required before sending an audit request"
    )
    save_raw_response: bool = Field(
        default=False,
        description="Whether to persist raw model response in audit output"
    )

    model_config = SettingsConfigDict(extra="forbid")


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    log_dir: str = Field(
        default="./logs",
        description="Directory for log files"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG/INFO/WARNING/ERROR)"
    )
    log_rotation: str = Field(
        default="100 MB",
        description="Log file rotation size"
    )
    log_retention: str = Field(
        default="30 days",
        description="Log file retention period"
    )
    log_format: str = Field(
        default="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        description="Log format string"
    )

    model_config = SettingsConfigDict(extra="forbid")


class Settings(BaseSettings):
    """Main application settings."""

    yolo: YOLOConfig = Field(default_factory=YOLOConfig)
    event: EventConfig = Field(default_factory=EventConfig)
    stream: StreamConfig = Field(default_factory=StreamConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    lane: LaneConfig = Field(default_factory=LaneConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="forbid",
    )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Settings":
        """Load settings from YAML file."""
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}
        return cls(**config_dict)


EventConfig.model_rebuild()


_settings: Settings = None


def get_settings() -> Settings:
    """Get global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

"""Large-model audit pipeline for event re-verification."""

import asyncio
import base64
import json
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, request

import cv2

from highway_incident_detection.config.settings import AuditConfig
from highway_incident_detection.detectors.event_types import (
    EventType,
    EventTypeLike,
    IncidentEvent,
    event_type_name,
)
from highway_incident_detection.utils.logger import get_logger

logger = get_logger(__name__)


DEFAULT_RISK_LEVELS = ("low", "medium", "high", "critical")
DEFAULT_SYSTEM_PROMPT = (
    "You are reviewing highway incident detections. "
    "Return strict JSON with keys: confirmed, risk_level, confidence, "
    "reason, recommended_action."
)
DEFAULT_USER_PROMPT_TEMPLATE = (
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
)
FALLBACK_RISK_BY_EVENT = {
    EventType.CONSTRUCTION: "medium",
    EventType.REVERSE_VEHICLE: "high",
    EventType.MOTORCYCLE: "medium",
    EventType.PEDESTRIAN: "high",
    EventType.ROAD_DEBRIS: "high",
    EventType.ABNORMAL_PARKING: "medium",
    EventType.CONGESTION: "medium",
}


@dataclass
class AuditResult:
    """Audit outcome returned by the secondary verifier."""

    confirmed: bool
    risk_level: str
    reason: str
    reviewer: str
    confidence: float
    recommended_action: str = ""
    raw_response: Optional[Dict[str, Any]] = None

    def to_dict(self, include_raw_response: bool = False) -> Dict[str, Any]:
        """Serialize the audit result."""
        result = {
            "confirmed": self.confirmed,
            "risk_level": self.risk_level,
            "reason": self.reason,
            "reviewer": self.reviewer,
            "confidence": self.confidence,
            "recommended_action": self.recommended_action,
        }
        if include_raw_response and self.raw_response is not None:
            result["raw_response"] = self.raw_response
        return result


class EventAuditor:
    """Audit events with a multimodal large model or a local fallback policy."""

    def __init__(self, config: AuditConfig):
        self.config = config

    @property
    def enabled(self) -> bool:
        """Whether the auditor should run."""
        return self.config.enabled

    async def audit_event(
        self,
        event: IncidentEvent,
        keyframe_path: str,
        video_path: str,
    ) -> AuditResult:
        """Audit a detected event and return a confirmation verdict."""
        if event.confidence < self.config.min_event_confidence:
            return self._fallback_result(
                event,
                reviewer="fallback_policy",
                reason=(
                    "Detector confidence below audit threshold, skipped remote audit "
                    "and applied local risk policy"
                ),
            )

        if (
            self.config.provider == "openai_compatible"
            and self.config.endpoint
            and self.config.api_key
        ):
            try:
                response = await asyncio.to_thread(
                    self._call_openai_compatible,
                    event,
                    keyframe_path,
                    video_path,
                )
                parsed = self._parse_response(response)
                if parsed is not None:
                    return parsed
            except Exception as exc:
                logger.warning(
                    f"Audit request failed, falling back to local policy: {exc}"
                )

        return self._fallback_result(
            event,
            reviewer="fallback_policy",
            reason="Remote audit unavailable, local risk policy applied",
        )

    def _call_openai_compatible(
        self,
        event: IncidentEvent,
        keyframe_path: str,
        video_path: str,
    ) -> Dict[str, Any]:
        """Call an OpenAI-compatible multimodal chat completion endpoint."""
        payload = self._build_openai_payload(event, keyframe_path, video_path)
        req = request.Request(
            self.config.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc

    def _build_openai_payload(
        self,
        event: IncidentEvent,
        keyframe_path: str,
        video_path: str,
    ) -> Dict[str, Any]:
        """Build the OpenAI-compatible request payload."""
        images = self._collect_images(keyframe_path, video_path)
        system_prompt, prompt = self._resolve_prompts(event)
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

        payload = {
            "model": self.config.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": content},
            ],
            "temperature": 0.1,
        }
        return payload

    def _resolve_prompts(self, event: IncidentEvent) -> Tuple[str, str]:
        """Resolve system and user prompts for one event."""
        event_override = self._get_event_prompt_override(event.event_type_value)
        system_prompt = self._load_prompt_text(
            prompt_path=getattr(event_override, "system_prompt_path", None),
            prompt_text=getattr(event_override, "system_prompt", None),
            fallback_path=self.config.system_prompt_path,
            fallback_text=self.config.system_prompt,
            default_text=DEFAULT_SYSTEM_PROMPT,
        )
        user_prompt = self._build_prompt(event, event_override=event_override)
        return system_prompt, user_prompt

    def _build_prompt(
        self,
        event: IncidentEvent,
        *,
        event_override: Optional[Any] = None,
    ) -> str:
        """Build the audit instruction prompt."""
        template = self._load_prompt_text(
            prompt_path=getattr(event_override, "user_prompt_path", None),
            prompt_text=getattr(event_override, "user_prompt_template", None),
            fallback_path=self.config.user_prompt_path,
            fallback_text=self.config.user_prompt_template,
            default_text=DEFAULT_USER_PROMPT_TEMPLATE,
        )
        context = self._prompt_context(event)
        try:
            return template.format_map(defaultdict(str, context))
        except Exception as exc:
            logger.warning(f"Failed to render custom audit user prompt, using default: {exc}")
            return DEFAULT_USER_PROMPT_TEMPLATE.format_map(defaultdict(str, context))

    def _get_event_prompt_override(self, event_type: str) -> Optional[Any]:
        """Return prompt override config for one event type if configured."""
        return self.config.event_prompt_overrides.get(event_type)

    def _load_prompt_text(
        self,
        *,
        prompt_path: Optional[str],
        prompt_text: Optional[str],
        fallback_path: Optional[str],
        fallback_text: Optional[str],
        default_text: str,
    ) -> str:
        """Load a prompt from file or inline config."""
        if prompt_path:
            try:
                text = Path(prompt_path).read_text(encoding="utf-8").strip()
                if text:
                    return text
            except Exception as exc:
                logger.warning(f"Failed to read audit prompt file {prompt_path}: {exc}")
        if prompt_text:
            return prompt_text.strip() or default_text
        if fallback_path:
            try:
                text = Path(fallback_path).read_text(encoding="utf-8").strip()
                if text:
                    return text
            except Exception as exc:
                logger.warning(f"Failed to read audit prompt file {fallback_path}: {exc}")
        return (fallback_text or default_text).strip() or default_text

    def _prompt_context(self, event: IncidentEvent) -> Dict[str, Any]:
        """Build template context for custom prompts."""
        metadata = event.metadata or {}
        return {
            "event_type": event.event_type_value,
            "confidence": event.confidence,
            "description": event.description,
            "stream_name": event.stream_name,
            "frame_number": event.frame_number,
            "timestamp": event.timestamp.isoformat(),
            "bbox": list(event.bbox) if event.bbox else None,
            "metadata": metadata,
            "metadata_json": json.dumps(metadata, ensure_ascii=False),
            "risk_levels": ", ".join(DEFAULT_RISK_LEVELS),
        }

    def _collect_images(self, keyframe_path: str, video_path: str) -> List[str]:
        """Collect keyframe and sampled video frames as base64 JPEG strings."""
        images: List[str] = []
        keyframe_b64 = self._encode_image_file(keyframe_path)
        if keyframe_b64:
            images.append(keyframe_b64)

        if self.config.sample_video_frames and video_path and len(images) < self.config.max_images:
            images.extend(
                self._sample_video_frames(
                    video_path,
                    max_samples=self.config.max_images - len(images),
                )
            )

        return images[: self.config.max_images]

    def _encode_image_file(self, image_path: str) -> Optional[str]:
        """Read an image file and convert it to base64 JPEG."""
        if not image_path:
            return None

        path = Path(image_path)
        if not path.exists():
            return None

        data = path.read_bytes()
        return base64.b64encode(data).decode("utf-8")

    def _sample_video_frames(self, video_path: str, max_samples: int) -> List[str]:
        """Sample a few representative frames from a generated event clip."""
        if max_samples <= 0:
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                return []

            sample_positions = sorted(
                {
                    min(total_frames - 1, int(total_frames * ratio))
                    for ratio in self._frame_ratios(max_samples)
                }
            )
            encoded: List[str] = []
            for frame_pos in sample_positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                ok, image_buffer = cv2.imencode(".jpg", frame)
                if ok:
                    encoded.append(base64.b64encode(image_buffer.tobytes()).decode("utf-8"))
            return encoded
        finally:
            cap.release()

    def _frame_ratios(self, max_samples: int) -> List[float]:
        """Return evenly distributed frame ratios."""
        if max_samples == 1:
            return [0.5]
        return [index / (max_samples - 1) for index in range(max_samples)]

    def _parse_response(self, response: Dict[str, Any]) -> Optional[AuditResult]:
        """Parse an OpenAI-compatible response into an audit result."""
        choices = response.get("choices") or []
        if not choices:
            return None

        content = choices[0].get("message", {}).get("content", "")
        parsed_content = self._extract_json(content)
        if parsed_content is None:
            return None

        risk_level = str(parsed_content.get("risk_level", "medium")).lower()
        if risk_level not in DEFAULT_RISK_LEVELS:
            risk_level = "medium"

        try:
            confidence = float(parsed_content.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5

        return AuditResult(
            confirmed=bool(parsed_content.get("confirmed", False)),
            risk_level=risk_level,
            reason=str(parsed_content.get("reason", "")).strip() or "No reason provided",
            reviewer=f"llm:{self.config.model_name}",
            confidence=max(0.0, min(1.0, confidence)),
            recommended_action=str(parsed_content.get("recommended_action", "")).strip(),
            raw_response=response if self.config.save_raw_response else None,
        )

    def _extract_json(self, content: Any) -> Optional[Dict[str, Any]]:
        """Extract the first JSON object from a model response."""
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            content = "\n".join(text_parts)

        if not isinstance(content, str):
            return None

        content = content.strip()
        if not content:
            return None

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None

    def _fallback_result(
        self,
        event: IncidentEvent,
        reviewer: str,
        reason: str,
    ) -> AuditResult:
        """Return a deterministic fallback decision when remote audit is unavailable."""
        baseline_risk = FALLBACK_RISK_BY_EVENT.get(event.event_type, "medium")
        confidence = max(0.3, min(0.95, event.confidence))
        confirmed = event.confidence >= self.config.min_event_confidence
        recommended_action = self._recommended_action(event.event_type, baseline_risk)
        return AuditResult(
            confirmed=confirmed,
            risk_level=baseline_risk,
            reason=reason,
            reviewer=reviewer,
            confidence=confidence,
            recommended_action=recommended_action,
        )

    def _recommended_action(self, event_type: EventTypeLike, risk_level: str) -> str:
        """Map event types and risk levels to operator actions."""
        event_name = event_type_name(event_type)
        if risk_level in {"high", "critical"}:
            return f"Escalate {event_name} to manual verification immediately"
        return f"Keep monitoring {event_name} and retain generated evidence"

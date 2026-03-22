from datetime import datetime

from highway_incident_detection.config.settings import EventConfig, Settings
from highway_incident_detection.detectors.event_types import EventType
from highway_incident_detection.detectors.event_detector import EventDetector
from highway_incident_detection.system import HighwayIncidentDetectionSystem
from highway_incident_detection.utils.tracker import TrackedObject


def test_settings_from_yaml_loads_yolov8x13_event_mapping(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
event:
  vehicle_class_ids: [0, 2, 3, 4, 5]
  construction_vehicle_class_ids: [1, 9, 10, 11]
  worker_class_id: 7
  construction_require_worker: false
  motorcycle_class_id: 6
  pedestrian_class_id: 8
  debris_class_ids: [12]
""".strip(),
        encoding="utf-8",
    )

    settings = Settings.from_yaml(str(config_path))

    assert settings.event.vehicle_class_ids == [0, 2, 3, 4, 5]
    assert settings.event.construction_vehicle_class_ids == [1, 9, 10, 11]
    assert settings.event.worker_class_id == 7
    assert settings.event.construction_require_worker is False
    assert settings.event.motorcycle_class_id == 6
    assert settings.event.pedestrian_class_id == 8
    assert settings.event.debris_class_ids == [12]


def test_settings_from_yaml_loads_multi_lane_zones(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
stream:
  streams:
    - name: "cam_1"
      url: "demo.mp4"
      enabled: true
      lane:
        enabled: true
        zones:
          - name: "near_right"
            roi_points: [[0, 10], [100, 10], [100, 100], [0, 100]]
            direction: "right"
          - name: "far_left"
            roi_points: [[200, 10], [300, 10], [300, 100], [200, 100]]
            direction: "left"
""".strip(),
        encoding="utf-8",
    )

    settings = Settings.from_yaml(str(config_path))

    lane = settings.stream.streams[0]["lane"]
    assert lane["enabled"] is True
    assert lane["zones"][0]["name"] == "near_right"
    assert lane["zones"][0]["direction"] == "right"
    assert lane["zones"][1]["name"] == "far_left"
    assert lane["zones"][1]["direction"] == "left"


def test_construction_detection_can_trigger_with_worker_only_when_configured():
    detector = EventDetector(
        EventConfig(
            construction_vehicle_class_ids=[1, 9, 10, 11],
            worker_class_id=7,
            construction_require_worker=False,
            construction_detection_frames=1,
        )
    )

    event = detector.detect_construction(
        detections=[(7, 0.91, (10.0, 20.0, 50.0, 80.0))],
        stream_name="cam_1",
        frame_number=1,
        timestamp=datetime.now(),
    )

    assert event is not None
    assert event.event_type == EventType.CONSTRUCTION
    assert event.metadata["construction_object_count"] == 0
    assert event.metadata["worker_count"] == 1


def test_abnormal_parking_uses_configured_vehicle_class_ids():
    detector = EventDetector(
        EventConfig(
            vehicle_class_ids=[0, 2, 3, 4, 5],
            parking_detection_frames=2,
            parking_speed_threshold=0.5,
            parking_position_threshold=5.0,
        )
    )
    timestamp = datetime.now()

    pedestrian_track = TrackedObject(
        track_id=1,
        class_id=8,
        bbox=(0.0, 0.0, 20.0, 40.0),
        center=(10.0, 20.0),
        velocity=(0.0, 0.0),
        frame_count=1,
        last_seen=1,
    )
    parked_vehicle_track = TrackedObject(
        track_id=2,
        class_id=0,
        bbox=(100.0, 100.0, 180.0, 160.0),
        center=(140.0, 130.0),
        velocity=(0.0, 0.0),
        frame_count=1,
        last_seen=1,
    )

    no_event = detector.detect_abnormal_parking(
        tracked_objects={1: pedestrian_track},
        stream_name="cam_1",
        frame_number=1,
        timestamp=timestamp,
    )
    assert no_event is None

    detector.detect_abnormal_parking(
        tracked_objects={2: parked_vehicle_track},
        stream_name="cam_1",
        frame_number=1,
        timestamp=timestamp,
    )
    event = detector.detect_abnormal_parking(
        tracked_objects={2: parked_vehicle_track},
        stream_name="cam_1",
        frame_number=2,
        timestamp=timestamp,
    )

    assert event is not None
    assert event.event_type == EventType.ABNORMAL_PARKING
    assert event.metadata["track_id"] == 2


def test_reverse_detection_respects_lane_direction():
    detector = EventDetector(
        EventConfig(
            vehicle_class_ids=[0],
            reverse_detection_frames=2,
            reverse_speed_threshold=-0.2,
        )
    )
    timestamp = datetime.now()
    tracked_vehicle = TrackedObject(
        track_id=3,
        class_id=0,
        bbox=(100.0, 100.0, 180.0, 160.0),
        center=(140.0, 130.0),
        velocity=(-1.0, 0.0),
        frame_count=2,
        last_seen=2,
    )

    no_event = detector.detect_reverse_vehicle(
        tracked_objects={3: tracked_vehicle},
        stream_name="cam_left",
        frame_number=1,
        timestamp=timestamp,
        lane_direction="left",
    )
    assert no_event is None

    event = detector.detect_reverse_vehicle(
        tracked_objects={3: tracked_vehicle},
        stream_name="cam_right",
        frame_number=1,
        timestamp=timestamp,
        lane_direction="right",
    )
    assert event is None

    event = detector.detect_reverse_vehicle(
        tracked_objects={3: tracked_vehicle},
        stream_name="cam_right",
        frame_number=2,
        timestamp=timestamp,
        lane_direction="right",
    )
    assert event is not None
    assert event.event_type == EventType.REVERSE_VEHICLE
    assert event.metadata["lane_direction"] == "right"


def test_normalize_lane_config_keeps_multiple_zones():
    normalized = HighwayIncidentDetectionSystem._normalize_lane_config(
        {
            "enabled": True,
            "zones": [
                {
                    "name": "near_right",
                    "roi_points": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "direction": "right",
                },
                {
                    "name": "far_left",
                    "roi_points": [[20, 0], [30, 0], [30, 10], [20, 10]],
                    "direction": "left",
                },
            ],
        }
    )

    assert normalized["enabled"] is True
    assert [zone["name"] for zone in normalized["zones"]] == ["near_right", "far_left"]
    assert [zone["direction"] for zone in normalized["zones"]] == ["right", "left"]


def test_process_frame_can_filter_events_by_stream_configuration():
    detector = EventDetector(
        EventConfig(
            pedestrian_class_id=8,
            debris_class_ids=[12],
        )
    )

    events = detector.process_frame(
        detections=[
            (8, 0.9, (10.0, 10.0, 30.0, 50.0)),
            (12, 0.8, (100.0, 100.0, 140.0, 140.0)),
        ],
        tracked_objects={},
        stream_name="cam_1",
        frame_number=1,
        timestamp=datetime.now(),
        enabled_event_types={"road_debris"},
    )

    assert len(events) == 1
    assert events[0].event_type == EventType.ROAD_DEBRIS


def test_system_initializes_per_stream_event_filters():
    system = HighwayIncidentDetectionSystem.__new__(HighwayIncidentDetectionSystem)
    system.settings = type(
        "SettingsStub",
        (),
        {
            "stream": type(
                "StreamStub",
                (),
                {
                    "streams": [
                        {
                            "name": "stream_1",
                            "disabled_event_types": ["reverse_vehicle"],
                        },
                        {
                            "name": "stream_2",
                            "enabled_event_types": ["road_debris"],
                        },
                    ]
                },
            )()
        },
    )()

    system._init_stream_event_filters()

    assert system.stream_event_filters["stream_1"]["disabled"] == {"reverse_vehicle"}
    assert system.stream_event_filters["stream_1"]["enabled"] is None
    assert system.stream_event_filters["stream_2"]["enabled"] == {"road_debris"}

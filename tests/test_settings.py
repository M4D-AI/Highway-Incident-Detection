from highway_incident_detection.config.settings import Settings


def test_settings_from_yaml_accepts_empty_file(tmp_path):
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    settings = Settings.from_yaml(str(config_path))

    assert settings.yolo.model_path == "yolov8n.pt"

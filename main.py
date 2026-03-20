"""Backward-compatible top-level entry point."""

from highway_incident_detection.system import HighwayIncidentDetectionSystem

__all__ = ["HighwayIncidentDetectionSystem"]


if __name__ == "__main__":
    from highway_incident_detection.cli import main

    main()

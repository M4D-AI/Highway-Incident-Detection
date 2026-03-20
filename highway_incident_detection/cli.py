"""Command-line interface for Highway Incident Detection System."""

import asyncio
import signal
import sys
import argparse

from highway_incident_detection.system import HighwayIncidentDetectionSystem


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Highway Incident Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )

    args = parser.parse_args()

    # Create system instance
    system = HighwayIncidentDetectionSystem(config_path=args.config)

    # Setup signal handlers
    def signal_handler(sig, frame):
        """Handle interrupt signals."""
        system.logger.info("Received interrupt signal, initiating shutdown...")
        system.running = False
        if system._shutdown_event:
            system._shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run system
    try:
        asyncio.run(system.run())
    except Exception as e:
        system.logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

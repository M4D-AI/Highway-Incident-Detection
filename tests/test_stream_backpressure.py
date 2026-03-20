import asyncio

from highway_incident_detection.config.settings import StreamConfig
from highway_incident_detection.stream.rtsp_stream import RTSPStreamProcessor


async def _noop_callback(stream_name, frame, frame_number, timestamp):
    await asyncio.sleep(0.05)


async def _create_pending_task():
    await asyncio.sleep(0.1)


def test_stream_processor_drops_frames_when_backpressured():
    processor = RTSPStreamProcessor(
        StreamConfig(
            streams=[{"name": "cam_1", "url": "rtsp://example.com", "enabled": True}],
            max_pending_frames=1,
            drop_frames_when_busy=True,
        ),
        frame_callback=_noop_callback,
    )

    async def run_check():
        task = asyncio.create_task(_create_pending_task())
        processor.callback_tasks["cam_1"] = {task}
        processor.dropped_frames["cam_1"] = 0
        accepted = await processor._wait_for_callback_slot("cam_1")
        await asyncio.gather(task, return_exceptions=True)
        return accepted

    accepted = asyncio.run(run_check())

    assert accepted is False
    assert processor.dropped_frames["cam_1"] == 1

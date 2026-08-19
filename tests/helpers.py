import numpy as np
from numpy.typing import NDArray
from visionapi.common_pb2 import MessageType
from visionapi.sae_pb2 import SaeMessage
from visionlib.pipeline.tools import jpeg

from frameanonymizer.config import (AnonymizationConfig,
                                    FrameAnonymizerConfig, RedisConfig)

FRAME_WIDTH = 128
FRAME_HEIGHT = 96


def make_config(class_ids=None, **kwargs) -> FrameAnonymizerConfig:
    return FrameAnonymizerConfig(
        log_level='WARNING',
        anonymization=AnonymizationConfig(
            class_ids=class_ids if class_ids is not None else [0],
            **kwargs
        ),
        redis=RedisConfig(
            stream_id='test_stream',
            input_stream_prefix='test_prefix'
        )
    )


def make_noise_image(width=FRAME_WIDTH, height=FRAME_HEIGHT) -> NDArray:
    '''High-contrast image, so that blurring provably changes pixels.'''
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def make_sae_msg(image: NDArray = None, detections=(), timestamp=1, as_jpeg=False) -> SaeMessage:
    '''Builds an SaeMessage. `detections` is a sequence of (class_id, (min_x, min_y, max_x, max_y)) tuples.'''
    sae_msg = SaeMessage()
    sae_msg.type = MessageType.SAE
    sae_msg.frame.source_id = 'test_source'
    sae_msg.frame.timestamp_utc_ms = timestamp

    if image is not None:
        sae_msg.frame.shape.height = image.shape[0]
        sae_msg.frame.shape.width = image.shape[1]
        sae_msg.frame.shape.channels = image.shape[2]
        if as_jpeg:
            sae_msg.frame.frame_data_jpeg = jpeg.encode(image, quality=95)
        else:
            sae_msg.frame.frame_data = image.tobytes()

    for class_id, (min_x, min_y, max_x, max_y) in detections:
        detection = sae_msg.detections.add()
        detection.class_id = class_id
        detection.confidence = 0.9
        detection.bounding_box.min_x = min_x
        detection.bounding_box.min_y = min_y
        detection.bounding_box.max_x = max_x
        detection.bounding_box.max_y = max_y

    return sae_msg


def make_sae_msg_bytes(**kwargs) -> bytes:
    return make_sae_msg(**kwargs).SerializeToString()

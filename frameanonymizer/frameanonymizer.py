import logging
import math
from typing import Any, Optional

import cv2
from numpy.typing import NDArray
from prometheus_client import Counter, Histogram, Summary
from visionapi.sae_pb2 import BoundingBox, SaeMessage
from visionlib.pipeline.tools import get_raw_frame_data, is_jpeg_frame, jpeg

from .config import FrameAnonymizerConfig

logging.basicConfig(format='%(asctime)s %(name)-15s %(levelname)-8s %(processName)-10s %(message)s')
logger = logging.getLogger(__name__)

# Bounding boxes smaller than this (in pixels) are not worth blurring and would only produce degenerate ROIs
MIN_BOX_SIZE_PX = 2

GET_DURATION = Histogram('frame_anonymizer_get_duration', 'The time it takes to deserialize the proto until returning the tranformed result as a serialized proto',
                         buckets=(0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25))
OBJECT_COUNTER = Counter('frame_anonymizer_object_counter', 'How many detections have been anonymized')
DROPPED_MESSAGE_COUNTER = Counter('frame_anonymizer_dropped_message_counter', 'How many messages have been dropped because their frame could not be anonymized')
PROTO_SERIALIZATION_DURATION = Summary('frame_anonymizer_proto_serialization_duration', 'The time it takes to create a serialized output proto')
PROTO_DESERIALIZATION_DURATION = Summary('frame_anonymizer_proto_deserialization_duration', 'The time it takes to deserialize an input proto')


def _kernel_size_for_sigma(sigma: float) -> int:
    '''Kernel width whose variance matches a Gaussian of `sigma` (a box of width k has variance (k^2 - 1) / 12).
       OpenCV requires an odd, positive kernel size, so the result is rounded to the nearest odd number.'''
    matching_width = math.sqrt(12 * sigma * sigma + 1)
    return max(3, int(round((matching_width - 1) / 2)) * 2 + 1)


class FrameAnonymizer:
    def __init__(self, config: FrameAnonymizerConfig) -> None:
        self.config = config
        logger.setLevel(self.config.log_level.value)

    def __call__(self, input_proto) -> Any:
        return self.get(input_proto)

    @GET_DURATION.time()
    def get(self, input_proto) -> Optional[bytes]:
        sae_msg = self._unpack_proto(input_proto)

        # get_raw_frame_data returns None for missing fields, but raises on malformed frame data
        try:
            frame = get_raw_frame_data(sae_msg.frame)
        except Exception:
            logger.exception('Error while decoding frame data')
            frame = None

        # We rather drop a message than let a non-anonymized frame pass through
        if frame is None:
            DROPPED_MESSAGE_COUNTER.inc()
            logger.warning(f'Could not retrieve frame data from message (source_id: {sae_msg.frame.source_id}, ts: {sae_msg.frame.timestamp_utc_ms}). Dropping message.')
            return None

        # Raw frames are backed by a read-only buffer (np.frombuffer), JPEG frames are not
        if not frame.flags.writeable:
            frame = frame.copy()

        for detection in sae_msg.detections:
            if detection.class_id not in self.config.anonymization.class_ids:
                continue
            if self._blur_bounding_box(frame, detection.bounding_box):
                OBJECT_COUNTER.inc()

        self._set_frame_data(sae_msg, frame)

        return self._pack_proto(sae_msg)

    def _blur_bounding_box(self, frame: NDArray, bounding_box: BoundingBox) -> bool:
        '''Blurs the image region covered by `bounding_box` (in place). Returns whether anything has been blurred.'''
        frame_height, frame_width = frame.shape[:2]

        # Bounding box coordinates are normalized to [0, 1] (see object-detector)
        min_x = max(0, int(bounding_box.min_x * frame_width))
        min_y = max(0, int(bounding_box.min_y * frame_height))
        max_x = min(frame_width, int(bounding_box.max_x * frame_width))
        max_y = min(frame_height, int(bounding_box.max_y * frame_height))

        box_width = max_x - min_x
        box_height = max_y - min_y

        if box_width < MIN_BOX_SIZE_PX or box_height < MIN_BOX_SIZE_PX:
            return False

        # Deriving sigma from the box size keeps the perceived blur strength independent of object size
        sigma = max(1.0, min(box_width, box_height) * self.config.anonymization.blur_strength)
        kernel_size = _kernel_size_for_sigma(sigma)

        # stackBlur runs on running sums, i.e. it costs the same no matter how large the kernel gets. That
        # matters here because sigma scales with the box (and therefore with the frame resolution): an
        # equivalent GaussianBlur is ~1400x slower on a 4K close-up, which would force blur_strength down.
        roi = frame[min_y:max_y, min_x:max_x]
        frame[min_y:max_y, min_x:max_x] = cv2.stackBlur(roi, (kernel_size, kernel_size))

        return True

    def _set_frame_data(self, sae_msg: SaeMessage, frame: NDArray) -> None:
        '''Writes the frame back into the message in the same format it arrived in.'''
        if is_jpeg_frame(sae_msg.frame):
            sae_msg.frame.frame_data_jpeg = jpeg.encode(frame, quality=self.config.anonymization.jpeg_quality)
        else:
            sae_msg.frame.frame_data = frame.tobytes()

    @PROTO_DESERIALIZATION_DURATION.time()
    def _unpack_proto(self, sae_message_bytes):
        sae_msg = SaeMessage()
        sae_msg.ParseFromString(sae_message_bytes)

        return sae_msg

    @PROTO_SERIALIZATION_DURATION.time()
    def _pack_proto(self, sae_msg: SaeMessage):
        return sae_msg.SerializeToString()

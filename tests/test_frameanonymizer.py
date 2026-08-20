import numpy as np
import pytest
from visionapi.sae_pb2 import SaeMessage
from visionlib.pipeline.tools import jpeg

from frameanonymizer.frameanonymizer import (FrameAnonymizer,
                                             _kernel_size_for_sigma)

from .helpers import (FRAME_HEIGHT, FRAME_WIDTH, make_config, make_noise_image,
                      make_sae_msg_bytes)

ANONYMIZED_CLASS = 0
UNTOUCHED_CLASS = 42

# Two non-overlapping boxes in the left / right half of the image
ANONYMIZED_BOX = (0.1, 0.1, 0.4, 0.9)
UNTOUCHED_BOX = (0.6, 0.1, 0.9, 0.9)


def _to_px(box):
    min_x, min_y, max_x, max_y = box
    return (int(min_x * FRAME_WIDTH), int(min_y * FRAME_HEIGHT),
            int(max_x * FRAME_WIDTH), int(max_y * FRAME_HEIGHT))


def _parse(output_bytes) -> SaeMessage:
    sae_msg = SaeMessage()
    sae_msg.ParseFromString(output_bytes)
    return sae_msg


def _raw_frame_to_image(sae_msg: SaeMessage):
    return np.frombuffer(sae_msg.frame.frame_data, dtype=np.uint8) \
        .reshape((sae_msg.frame.shape.height, sae_msg.frame.shape.width, sae_msg.frame.shape.channels))


@pytest.fixture
def anonymizer():
    return FrameAnonymizer(make_config(class_ids=[ANONYMIZED_CLASS]))


def test_raw_frame_only_configured_class_is_blurred(anonymizer):
    image = make_noise_image()
    input_bytes = make_sae_msg_bytes(
        image=image,
        detections=[(ANONYMIZED_CLASS, ANONYMIZED_BOX), (UNTOUCHED_CLASS, UNTOUCHED_BOX)]
    )

    # This also covers the read-only np.frombuffer buffer that get_raw_frame_data returns for raw frames
    output = _parse(anonymizer.get(input_bytes))
    output_image = _raw_frame_to_image(output)

    assert output_image.shape == image.shape

    min_x, min_y, max_x, max_y = _to_px(ANONYMIZED_BOX)
    assert not np.array_equal(output_image[min_y:max_y, min_x:max_x], image[min_y:max_y, min_x:max_x])

    min_x, min_y, max_x, max_y = _to_px(UNTOUCHED_BOX)
    assert np.array_equal(output_image[min_y:max_y, min_x:max_x], image[min_y:max_y, min_x:max_x])

    # Everything to the right of the anonymized box has to be untouched as well
    _, _, anonymized_max_x, _ = _to_px(ANONYMIZED_BOX)
    assert np.array_equal(output_image[:, anonymized_max_x:], image[:, anonymized_max_x:])


def test_raw_frame_stays_raw(anonymizer):
    image = make_noise_image()
    input_bytes = make_sae_msg_bytes(image=image, detections=[(ANONYMIZED_CLASS, ANONYMIZED_BOX)])

    output = _parse(anonymizer.get(input_bytes))

    assert output.frame.frame_data != b''
    assert output.frame.frame_data_jpeg == b''


def test_jpeg_frame_is_blurred_and_stays_jpeg(anonymizer):
    image = make_noise_image()
    input_bytes = make_sae_msg_bytes(image=image, detections=[(ANONYMIZED_CLASS, ANONYMIZED_BOX)], as_jpeg=True)

    output = _parse(anonymizer.get(input_bytes))

    assert output.frame.frame_data_jpeg != b''
    assert output.frame.frame_data == b''

    output_image = jpeg.decode(output.frame.frame_data_jpeg)
    min_x, min_y, max_x, max_y = _to_px(ANONYMIZED_BOX)

    # JPEG is lossy, so we cannot compare pixels - a blurred region has significantly less variance instead
    assert np.var(output_image[min_y:max_y, min_x:max_x]) < np.var(image[min_y:max_y, min_x:max_x]) / 2


def test_detections_are_preserved(anonymizer):
    input_bytes = make_sae_msg_bytes(
        image=make_noise_image(),
        detections=[(ANONYMIZED_CLASS, ANONYMIZED_BOX), (UNTOUCHED_CLASS, UNTOUCHED_BOX)]
    )

    output = _parse(anonymizer.get(input_bytes))

    assert len(output.detections) == 2
    assert output.detections[0].class_id == ANONYMIZED_CLASS
    assert output.detections[0].bounding_box.min_x == pytest.approx(ANONYMIZED_BOX[0])


def test_message_without_frame_data_is_dropped(anonymizer):
    assert anonymizer.get(make_sae_msg_bytes(detections=[(ANONYMIZED_CLASS, ANONYMIZED_BOX)])) is None


def test_message_with_undecodable_frame_data_is_dropped(anonymizer):
    sae_msg = SaeMessage()
    sae_msg.frame.source_id = 'test_source'
    sae_msg.frame.timestamp_utc_ms = 1
    sae_msg.frame.shape.height = FRAME_HEIGHT
    sae_msg.frame.shape.width = FRAME_WIDTH
    sae_msg.frame.shape.channels = 3
    sae_msg.frame.frame_data = b'not an image'

    assert anonymizer.get(sae_msg.SerializeToString()) is None


@pytest.mark.parametrize('box', [
    (0.0, 0.0, 0.0, 0.0),       # empty
    (0.5, 0.5, 0.5001, 0.5001),  # sub-pixel
    (0.4, 0.4, 0.2, 0.2),       # inverted
    (1.5, 1.5, 2.0, 2.0),       # fully outside
    (-0.5, -0.5, 0.2, 0.2),     # partially outside
])
def test_degenerate_boxes_do_not_crash(anonymizer, box):
    image = make_noise_image()

    output = anonymizer.get(make_sae_msg_bytes(image=image, detections=[(ANONYMIZED_CLASS, box)]))

    assert output is not None
    assert _raw_frame_to_image(_parse(output)).shape == image.shape


@pytest.mark.parametrize('sigma', [0.1, 1.0, 5.0, 30.0, 180.0])
def test_kernel_size_is_odd_and_variance_matched(sigma):
    kernel_size = _kernel_size_for_sigma(sigma)

    # OpenCV requires an odd, positive kernel size
    assert kernel_size >= 3
    assert kernel_size % 2 == 1

    # A box of width k has variance (k^2 - 1) / 12, which should match the Gaussian it replaces.
    # Only checked for sigmas where rounding to an odd kernel size is not the dominant error.
    if sigma >= 5.0:
        assert (kernel_size ** 2 - 1) / 12 == pytest.approx(sigma ** 2, rel=0.05)


def test_stronger_blur_leaves_less_detail(anonymizer):
    image = make_noise_image()
    input_bytes = make_sae_msg_bytes(image=image, detections=[(ANONYMIZED_CLASS, ANONYMIZED_BOX)])
    min_x, min_y, max_x, max_y = _to_px(ANONYMIZED_BOX)

    def residual_detail(blur_strength):
        anonymizer.config.anonymization.blur_strength = blur_strength
        output_image = _raw_frame_to_image(_parse(anonymizer.get(input_bytes)))
        return np.var(output_image[min_y:max_y, min_x:max_x].astype(np.float64))

    assert residual_detail(0.5) < residual_detail(0.05) < np.var(image[min_y:max_y, min_x:max_x].astype(np.float64))


def test_message_without_detections_is_forwarded(anonymizer):
    image = make_noise_image()

    output = _parse(anonymizer.get(make_sae_msg_bytes(image=image)))

    assert np.array_equal(_raw_frame_to_image(output), image)

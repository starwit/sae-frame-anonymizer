from unittest.mock import patch

import numpy as np
import pytest
from visionapi.sae_pb2 import SaeMessage

from frameanonymizer.stage import run_stage

from .helpers import make_config, make_noise_image, make_sae_msg_bytes

ANONYMIZED_CLASS = 0
BOX = (0.1, 0.1, 0.4, 0.9)


@pytest.fixture(autouse=True)
def disable_prometheus():
    # We don't want to start the Prometheus server during tests
    with patch('frameanonymizer.stage.start_http_server'):
        yield

@pytest.fixture
def set_config():
    with patch('frameanonymizer.stage.FrameAnonymizerConfig') as mock_config:
        def _make_mock_config(stream_id: str, input_stream_prefix: str):
            config = make_config(class_ids=[ANONYMIZED_CLASS])
            config.redis.stream_id = stream_id
            config.redis.input_stream_prefix = input_stream_prefix
            mock_config.return_value = config
        yield _make_mock_config

@pytest.fixture
def redis_publisher_mock():
    with patch('frameanonymizer.stage.ValkeyPublisher') as mock_publisher:
        yield mock_publisher.return_value.__enter__.return_value

@pytest.fixture
def inject_consumer_messages():
    with patch('frameanonymizer.stage.ValkeyConsumer') as mock_consumer:
        def _inject_messages(messages):
            mock_consumer.return_value.__enter__.return_value.return_value.__iter__.return_value = iter(messages)
        yield _inject_messages

def test_messages_are_anonymized_and_published(set_config, redis_publisher_mock, inject_consumer_messages):
    set_config(stream_id='test_stream', input_stream_prefix='test_prefix')

    image = make_noise_image()
    inject_consumer_messages([
        ('test_prefix:test_stream', _make_msg_bytes(image, 1)),
        ('test_prefix:test_stream', _make_msg_bytes(image, 2)),
    ])

    # Run the stage (this will process the injected messages)
    run_stage()

    # Verify that messages were published (anonymized, i.e. not byte-identical to the input)
    assert redis_publisher_mock.call_count == 2
    for call, expected_timestamp in zip(redis_publisher_mock.call_args_list, (1, 2)):
        stream_key, proto_data = call.args
        assert stream_key == 'frameanonymizer:test_stream'

        published_msg = SaeMessage()
        published_msg.ParseFromString(proto_data)
        assert published_msg.frame.timestamp_utc_ms == expected_timestamp
        assert published_msg.frame.frame_data != image.tobytes()

def test_unanonymizable_messages_are_not_published(set_config, redis_publisher_mock, inject_consumer_messages):
    set_config(stream_id='test_stream', input_stream_prefix='test_prefix')

    # No frame data -> nothing to anonymize -> must not be forwarded
    inject_consumer_messages([
        ('test_prefix:test_stream', make_sae_msg_bytes(timestamp=1)),
    ])

    run_stage()

    assert redis_publisher_mock.call_count == 0

def _make_msg_bytes(image: np.ndarray, timestamp: int) -> bytes:
    return make_sae_msg_bytes(image=image, detections=[(ANONYMIZED_CLASS, BOX)], timestamp=timestamp)

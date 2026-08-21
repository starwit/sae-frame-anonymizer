import pytest
from pydantic import ValidationError

from frameanonymizer.config import (AnonymizationConfig,
                                    FrameAnonymizerConfig, RedisConfig)


def test_incomplete_config():
    with pytest.raises(ValidationError):
        FrameAnonymizerConfig(
            log_level="INFO",
            anonymization=AnonymizationConfig(class_ids=[0]),
            redis=RedisConfig(
                host="localhost",
                port=6379
                # Missing stream_id and input_stream_prefix
            )
        )

def test_missing_anonymization_config():
    with pytest.raises(ValidationError):
        FrameAnonymizerConfig(
            log_level="INFO",
            redis=RedisConfig(
                host="localhost",
                port=6379,
                stream_id="stream1",
                input_stream_prefix="frameanonymizer_input"
            )
        )

def test_empty_class_ids():
    with pytest.raises(ValidationError):
        AnonymizationConfig(class_ids=[])

@pytest.mark.parametrize('kwargs', [
    {'class_ids': [0], 'blur_strength': 0},
    {'class_ids': [0], 'blur_strength': 1.5},
    {'class_ids': [0], 'jpeg_quality': 0},
    {'class_ids': [0], 'jpeg_quality': 101},
])
def test_invalid_anonymization_values(kwargs):
    with pytest.raises(ValidationError):
        AnonymizationConfig(**kwargs)

def test_complete_config():
    config = FrameAnonymizerConfig(
        log_level="INFO",
        anonymization=AnonymizationConfig(
            class_ids=[0, 2],
            blur_strength=0.2,
            jpeg_quality=90
        ),
        redis=RedisConfig(
            host="localhost",
            port=6379,
            stream_id="stream1",
            input_stream_prefix="frameanonymizer_input"
        ),
        prometheus_port=9000
    )

    assert config.log_level.name == "INFO"
    assert config.anonymization.class_ids == [0, 2]
    assert config.anonymization.blur_strength == 0.2
    assert config.anonymization.jpeg_quality == 90
    assert config.redis.host == "localhost"
    assert config.redis.port == 6379
    assert config.redis.stream_id == "stream1"
    assert config.redis.input_stream_prefix == "frameanonymizer_input"
    assert config.prometheus_port == 9000

def test_anonymization_defaults():
    config = AnonymizationConfig(class_ids=[0])

    assert config.blur_strength == 0.15
    assert config.jpeg_quality == 85

import os
from typing import List

from pydantic import BaseModel, Field
from pydantic_settings import (BaseSettings, SettingsConfigDict,
                               YamlConfigSettingsSource)
from typing_extensions import Annotated
from visionlib.pipeline.settings import LogLevel


class AnonymizationConfig(BaseModel):
    # The class ids whose detections should be anonymized (no default on purpose, this has to be an explicit decision)
    class_ids: Annotated[List[int], Field(min_length=1)]
    # Gaussian sigma as a fraction of the (smaller) bounding box dimension, i.e. blur strength is independent of object size
    blur_strength: Annotated[float, Field(gt=0, le=1)] = 0.15
    # Only relevant for frames that arrive as JPEG (they are re-encoded after anonymization)
    jpeg_quality: Annotated[int, Field(ge=1, le=100)] = 85


class RedisConfig(BaseModel):
    host: str = 'localhost'
    port: Annotated[int, Field(ge=1, le=65536)] = 6379
    stream_id: str
    input_stream_prefix: str
    output_stream_prefix: str = 'frameanonymizer'

class FrameAnonymizerConfig(BaseSettings):
    log_level: LogLevel = LogLevel.WARNING
    anonymization: AnonymizationConfig
    redis: RedisConfig
    prometheus_port: Annotated[int, Field(ge=1024, le=65536)] = 8000

    model_config = SettingsConfigDict(env_nested_delimiter='__')

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        YAML_LOCATION = os.environ.get('SETTINGS_FILE', 'settings.yaml')
        return (init_settings, env_settings, YamlConfigSettingsSource(settings_cls, yaml_file=YAML_LOCATION), file_secret_settings)
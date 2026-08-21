# SAE frame-anonymizer

Reads SaeMessages from pipeline, anonymizes the actual frame data and publishes the anoymized message.

## Anonymization
The first draft uses opencv to blur entire bounding boxes of detected objects which may contain privacy sensitive information (i.e. COCO "person", "car", etc.). The class ids that should be anonymized can be configured.

Blurring uses `cv2.stackBlur`, which approximates a Gaussian but runs on running sums and therefore costs the same regardless of kernel size. That matters because the blur radius scales with the bounding box (and thus with the frame resolution) — an equivalent `cv2.GaussianBlur` is roughly 1400x slower on a 4K close-up.

Frames are written back in the format they arrived in (JPEG in → JPEG out, uncompressed in → uncompressed out). The stage fails closed: if a frame cannot be decoded (or a message carries no frame data at all), the message is dropped instead of being forwarded un-anonymized. Dropped messages are counted in `frame_anonymizer_dropped_message_counter`.

## Configuration
See [settings.template.yaml](settings.template.yaml) for a complete example.

| Setting | Default | Description |
| --- | --- | --- |
| `anonymization.class_ids` | *required* | Class ids whose detections are blurred. Detections of any other class are left untouched. |
| `anonymization.blur_strength` | `0.15` | Blur radius as a fraction of the smaller bounding box dimension, so the perceived blur strength does not depend on how large/close an object is (or on the frame resolution). Raise it for stronger anonymization. |
| `anonymization.jpeg_quality` | `90` | Quality used to re-encode frames that arrived as JPEG. Ignored for uncompressed frames. |
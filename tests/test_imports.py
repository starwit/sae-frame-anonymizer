import pytest

def test_frameanonymizer_import():
    try:
        from frameanonymizer.frameanonymizer import FrameAnonymizer
    except ImportError as e:
        pytest.fail(f"Failed to import FrameAnonymizer: {e}")

    assert FrameAnonymizer is not None, "FrameAnonymizer should be imported successfully"
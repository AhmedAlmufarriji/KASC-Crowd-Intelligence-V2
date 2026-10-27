"""Optional CV tests run only if the optional OpenCV extra is installed."""
import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
from kasc_v2.vision import analyze_photo


def test_detect_green_region_in_artificial_stadium():
    frame = np.full((360, 360, 3), (26, 37, 42), dtype=np.uint8)
    cv2.ellipse(frame, (180, 180), (140, 145), 0, 0, 360, (198, 205, 214), -1)
    cv2.rectangle(frame, (122, 106), (238, 252), (51, 126, 46), -1)
    ok, image = cv2.imencode(".png", frame)
    assert ok
    result = analyze_photo(image.tobytes())
    assert result["pitch_bbox"] is not None
    assert result["pitch_fraction"] > 0.05
    assert result["edge_pixel_count"] > 0
    assert result["overlay_png_base64"]


def test_reject_invalid_input():
    with pytest.raises(ValueError):
        analyze_photo(b"not an image")

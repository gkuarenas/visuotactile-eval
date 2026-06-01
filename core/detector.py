import cv2
import numpy as np
from scipy.ndimage import maximum_filter


def default_params() -> dict:
    return {
        "thresh": 100,
        "erode": True,
        "open": True,
        "dilate": True,
        "log_ksize": 61,
        "log_sigma": 20.0,
    }


def preprocess(frame_gray: np.ndarray, params: dict) -> np.ndarray:
    _, img = cv2.threshold(frame_gray, params["thresh"], 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), dtype=np.uint8)
    if params["erode"]:
        img = cv2.erode(img, kernel, iterations=1)
    if params["open"]:
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    if params["dilate"]:
        img = cv2.dilate(img, kernel, iterations=1)
    return img


def build_log_kernel(ksize: int, sigma: float) -> np.ndarray:
    assert ksize % 2 == 1, "ksize must be odd"
    ax = np.linspace(-(ksize // 2), ksize // 2, ksize)
    xx, yy = np.meshgrid(ax, ax)
    r2 = xx ** 2 + yy ** 2
    # Positive centre: responds maximally at centres of BRIGHT blobs
    kernel = (1.0 - r2 / (2.0 * sigma ** 2)) * np.exp(-r2 / (2.0 * sigma ** 2))
    total = np.abs(kernel).sum()
    if total > 0:
        kernel /= total
    return kernel


def _estimate_area(img: np.ndarray, cx: int, cy: int, radius: int = 12) -> float:
    x0 = max(0, cx - radius)
    x1 = min(img.shape[1], cx + radius)
    y0 = max(0, cy - radius)
    y1 = min(img.shape[0], cy + radius)
    patch = img[y0:y1, x0:x1]
    return float(np.count_nonzero(patch))


def detect(
    gray: np.ndarray,
    preprocessed: np.ndarray,
    params: dict,
) -> list[tuple[float, float, float]]:
    ksize = params["log_ksize"]
    sigma = params["log_sigma"]

    kernel = build_log_kernel(ksize, sigma)
    # Apply to grayscale so the intensity gradient at blob centres is visible
    response = cv2.filter2D(gray.astype(np.float32), -1, kernel)

    local_max = response == maximum_filter(response, size=ksize)
    threshold = response.mean() + response.std()
    local_max &= response > threshold
    # Restrict to pixels inside white (marker) regions of the binary mask
    local_max &= preprocessed > 0

    ys, xs = np.where(local_max)

    h, w = gray.shape[:2]
    detections: list[tuple[float, float, float]] = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        if 0 <= x < w and 0 <= y < h:
            area = _estimate_area(preprocessed, x, y)
            detections.append((float(x), float(y), float(area)))

    return detections

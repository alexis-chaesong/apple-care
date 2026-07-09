"""
ros_image_utils.py
====================
sensor_msgs/Image <-> numpy 배열 변환. **설치된 numpy 메이저 버전**에 따라
구현을 분기한다 (ROS_DISTRO가 아님 - 아래 이유 참고).

배경: apt `python3-cv-bridge`의 `cv_bridge_boost` C 확장은, 그 패키지가
빌드될 당시 시스템에 있던 numpy의 C-ABI에 맞춰 컴파일되어 고정된다. numpy
2.0(2024년 중반 출시)에서 내부 구조가 바뀌어서, numpy 1.x ABI로 빌드된
확장을 numpy 2.x 런타임에서 쓰면 `AttributeError: _ARRAY_API not found`로
즉시 segfault난다. 이 프로젝트는 다른 워크스페이스와 numpy 2.x를 공유해야
해서 numpy 자체를 내릴 수는 없다.

실제 실패 조건은 배포판이 아니라 "각자 머신에 numpy2가 깔려 있는가"였다 -
같은 코드가 배포판과 무관하게 사람마다 다르게 재현/미재현됐다. 그래서
**"지금 import된 numpy가 메이저 2 이상인가"**로 직접 분기한다. numpy 2.0
출시 이전에 빌드된 cv_bridge는 전부 numpy1 ABI이므로, 이 체크 하나로 위
문제를 해결한다.
(openclose.py가 pymodbus 2.x/3.x 차이를 흡수하는 것과 같은 패턴 - "배포판"이
아니라 "실제로 설치된 라이브러리 버전"으로 분기.)
"""

import numpy as np
from sensor_msgs.msg import Image

# (numpy dtype, 채널 수)
_ENCODING_TO_DTYPE_CHANNELS = {
    "mono8": (np.uint8, 1),
    "8UC1": (np.uint8, 1),
    "bgr8": (np.uint8, 3),
    "rgb8": (np.uint8, 3),
    "bgra8": (np.uint8, 4),
    "rgba8": (np.uint8, 4),
    "mono16": (np.uint16, 1),
    "16UC1": (np.uint16, 1),
    "32FC1": (np.float32, 1),
}

# 채널 순서만 뒤집으면 되는 인코딩 쌍 (알파 채널은 그대로 유지)
_CHANNEL_SWAP_PAIRS = {
    ("bgr8", "rgb8"), ("rgb8", "bgr8"),
    ("bgra8", "rgba8"), ("rgba8", "bgra8"),
}


def _np_imgmsg_to_cv2(msg, desired_encoding="passthrough"):
    """sensor_msgs/Image -> numpy 배열. cv_bridge.imgmsg_to_cv2()의 축소 버전."""
    if msg.encoding not in _ENCODING_TO_DTYPE_CHANNELS:
        raise ValueError(f"지원하지 않는 encoding: {msg.encoding}")
    dtype, channels = _ENCODING_TO_DTYPE_CHANNELS[msg.encoding]
    dtype = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")

    row_items = msg.step // dtype.itemsize
    raw = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, row_items)
    raw = raw[:, : msg.width * channels]
    img = raw.reshape(msg.height, msg.width, channels) if channels > 1 else raw.reshape(msg.height, msg.width)

    if desired_encoding in ("passthrough", msg.encoding):
        return img
    if (msg.encoding, desired_encoding) in _CHANNEL_SWAP_PAIRS:
        return img[:, :, [2, 1, 0, *range(3, channels)]]
    raise ValueError(f"지원하지 않는 변환: {msg.encoding} -> {desired_encoding}")


def _np_cv2_to_imgmsg(cv_image, encoding="bgr8"):
    """numpy 배열 -> sensor_msgs/Image. cv_bridge.cv2_to_imgmsg()의 축소 버전."""
    if encoding not in _ENCODING_TO_DTYPE_CHANNELS:
        raise ValueError(f"지원하지 않는 encoding: {encoding}")
    dtype, channels = _ENCODING_TO_DTYPE_CHANNELS[encoding]
    arr = np.ascontiguousarray(cv_image, dtype=dtype)

    msg = Image()
    msg.height, msg.width = arr.shape[0], arr.shape[1]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = arr.shape[1] * channels * np.dtype(dtype).itemsize
    msg.data = arr.tobytes()
    return msg


_NUMPY_MAJOR_VERSION = int(np.__version__.split(".")[0])

if _NUMPY_MAJOR_VERSION >= 2:
    # numpy2 ABI 문제를 피해 cv_bridge 자체를 우회 (파일 상단 docstring 참고)
    imgmsg_to_cv2 = _np_imgmsg_to_cv2
    cv2_to_imgmsg = _np_cv2_to_imgmsg
else:
    from cv_bridge import CvBridge

    _bridge = CvBridge()

    def imgmsg_to_cv2(msg, desired_encoding="passthrough"):
        return _bridge.imgmsg_to_cv2(msg, desired_encoding=desired_encoding)

    def cv2_to_imgmsg(cv_image, encoding="bgr8"):
        return _bridge.cv2_to_imgmsg(cv_image, encoding=encoding)

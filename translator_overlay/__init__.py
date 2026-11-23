import os

# MSMF hardware transforms add ~20 s to opening a UVC card; OpenCV reads this at import, so set it before importing cv2.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

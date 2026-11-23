"""Windows only: stand-in for a game window. Plays a video in a window at its native frame rate.
python tools/test_player.py tests/cropped.mp4 [seconds] [title]"""
import sys
import time

import cv2

path = sys.argv[1]
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 25
title = sys.argv[3] if len(sys.argv) > 3 else "TRO test player"
cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
cv2.moveWindow(title, 20, 10)
t0, i = time.perf_counter(), 0
while i < secs * fps:
    ok, f = cap.read()
    if not ok:
        break
    cv2.imshow(title, f)
    cv2.waitKey(1)
    i += 1
    time.sleep(max(0, t0 + i / fps - time.perf_counter()))
cv2.destroyAllWindows()

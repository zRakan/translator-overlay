"""Windows only: list capture devices (capture cards, webcams, virtual cameras) with the index to use in cam:<index>.
python tools/list_cameras.py"""
import subprocess
import sys

try:
    from cv2_enumerate_cameras import enumerate_cameras
    import cv2
    for b, name in ((cv2.CAP_DSHOW, "dshow"), (cv2.CAP_MSMF, "msmf")):
        for c in enumerate_cameras(b):
            print(f"cam:{c.index}  [{name}]  {c.name}   (vid={getattr(c, 'vid', '?')} pid={getattr(c, 'pid', '?')})")
except ImportError:
    print("(pip install cv2-enumerate-cameras for DirectShow/MSMF indexes; showing Windows' device list instead)", file=sys.stderr)
    ps = "Get-PnpDevice -Class Camera,Image -ErrorAction SilentlyContinue | Where-Object Status -eq 'OK' | Select-Object -ExpandProperty FriendlyName"
    print(subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout.strip() or "no capture devices found")

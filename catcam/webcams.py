"""The Mac's cameras by name, numbered as OpenCV numbers them, and which one to use.

OpenCV numbers cameras by their unique ID rather than by any preference, so camera
0 can be an iPhone (Continuity Camera) lying nearby, which opens but may send
nothing, or OBS's virtual camera, which is where Cat Cam's own picture goes.
"""

import sys
from typing import NamedTuple

VIRTUAL = ("OBS Virtual Camera",)  # Cat Cam's own output: never read from it


class Webcam(NamedTuple):
    index: int  # OpenCV's number for it
    name: str
    model: str
    built_in: bool


def webcams() -> list[Webcam]:
    """The cameras in OpenCV's order (see orderedVideoCaptureDevices in its
    cap_avfoundation_mac.mm); empty where they can't be listed."""
    if sys.platform != "darwin":
        return []
    import AVFoundation as AV

    devices = [*AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo),
               *AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeMuxed)]
    devices.sort(key=lambda d: d.uniqueID())
    return [Webcam(i, str(d.localizedName()), str(d.modelID() or ""),
                   d.deviceType() == AV.AVCaptureDeviceTypeBuiltInWideAngleCamera)
            for i, d in enumerate(devices)]


def preference(cam: Webcam) -> int | None:
    """Lower is better: the Mac's own camera, then other webcams, then an iPhone or
    iPad, which may be asleep or out of reach. None for a camera never to read."""
    if cam.name in VIRTUAL:
        return None
    if cam.built_in:
        return 0
    if cam.model.startswith(("iPhone", "iPad")):
        return 2
    return 1


def choose(cams: list[Webcam]) -> Webcam | None:
    """The camera to use when none is asked for, if any can be."""
    usable = [c for c in cams if preference(c) is not None]
    return min(usable, key=preference, default=None)

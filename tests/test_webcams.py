from catcam.webcams import Webcam, choose

MAC = Webcam(1, "MacBook Pro Camera", "MacBook Pro Camera", True)
PHONE = Webcam(0, "Marci Camera", "iPhone16,2", False)
OBS = Webcam(0, "OBS Virtual Camera", "OBS Camera Extension", False)
LOGITECH = Webcam(2, "C920 HD Pro Webcam", "UVC Camera VendorID_1133 ProductID_2093", False)


def test_the_macs_own_camera_comes_first():
    assert choose([PHONE, MAC, LOGITECH]) == MAC


def test_a_webcam_before_an_iphone():
    assert choose([PHONE, LOGITECH]) == LOGITECH
    assert choose([PHONE]) == PHONE


def test_never_cat_cams_own_output():
    assert choose([OBS, LOGITECH]) == LOGITECH
    assert choose([OBS]) is None
    assert choose([]) is None

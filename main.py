from typing import Tuple

import numpy as np
import cv2
from skimage.feature import corner_peaks


def compute_grads(img1: np.ndarray, img2: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

    grad_x = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3) / 255.0
    grad_y = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3) / 255.0

    if img2 is not None:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        grad_t = (img2 - img1) / 255.0
    else:
        grad_t = None

    return grad_x, grad_y, grad_t



def lucas_kanade(grad_x: np.ndarray, grad_y: np.ndarray, grad_t: np.ndarray, windows_size: int = 3) -> Tuple[
    np.ndarray, np.ndarray]:
    pass


def find_feature_points(grad_x: np.ndarray, grad_y: np.ndarray, windows_size: int = 3, kappa: float = 0.05,
                               sigma: float = 1.0, thresh: float = 0.02) -> np.ndarray:

    # Structure tensor + gaussian filter
    Ixx = cv2.GaussianBlur(grad_x ** 2, ksize=(windows_size, windows_size), sigmaX=sigma, sigmaY=sigma)
    Iyy = cv2.GaussianBlur(grad_y ** 2, ksize=(windows_size, windows_size), sigmaX=sigma, sigmaY=sigma)
    Ixy = cv2.GaussianBlur(grad_x * grad_y, ksize=(windows_size, windows_size), sigmaX=sigma, sigmaY=sigma)


    det = Ixx * Iyy - Ixy ** 2
    trace = Ixx + Iyy
    response = det - kappa * trace ** 2

    # Non maximum suppression
    response = corner_peaks(response, indices=True, min_distance=3, threshold_rel=thresh)

    return response

def run_feature_detector() -> None:
    cv2.namedWindow('preview')
    cv2.namedWindow('sliders')

    cv2.setWindowProperty('preview', cv2.WND_PROP_TOPMOST, 1)
    cv2.setWindowProperty('sliders', cv2.WND_PROP_TOPMOST, 1)

    vc = cv2.VideoCapture(0)

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    '''thresh = 0.1

    def callback_thresh(val: float):
        nonlocal thresh
        thresh = val / 1000 + 1e-10

    cv2.createTrackbar('trackbar_name', 'sliders', 0, 100, callback_thresh)'''

    while rval:
        rval, frame = vc.read()
        grad_x, grad_y, _ = compute_grads(frame)
        points = find_feature_points(grad_x, grad_y)
        #print(thresh)

        # Draw all detected points
        for p in points:
            cv2.circle(frame, (p[1], p[0]), radius=3, color=(0, 0, 255))

        cv2.imshow('preview', frame)

        rval, frame = vc.read()
        key = cv2.waitKey(30)
        if key == 27:  # exit on ESC
            break

    vc.release()
    cv2.destroyWindow("preview")


if __name__ == '__main__':
    run_feature_detector()

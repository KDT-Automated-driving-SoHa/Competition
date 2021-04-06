import cv2 as cv
import numpy as np
from os.path import join as PathJoin


def setROI(frame, tb):
    x, y = tb.getValue("x"), tb.getValue("y")
    width, height = tb.getValue("width"), tb.getValue("height")

    return frame[y:y+height, x:x+width]


def adaptiveThreshold(frame, tb):
    if frame.ndim == 3:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    else:
        gray = frame

    ksize = tb.getValue("Gaussian_ksize")
    blur = cv.GaussianBlur(gray, (ksize, ksize), sigmaX=0)
    
    C = tb.getValue("C")
    blockSize = tb.getValue("blockSize")
    return cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize, C)
    


def Canny(gray, tb):
    if gray.ndim == 3:
        gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)

    blur = cv.GaussianBlur(gray, (5, 5), sigmaX=0)

    threshold = tb.getValue("threshold")
    ratio = tb.getValue("ratio")
    L2gradient = tb.getValue("L2gradient")

    return cv.Canny(gray, threshold, threshold*ratio, L2gradient)


def HoughLinesP(canny, tb):
    rho = tb.getValue("rho") / 10.0
    threshold = tb.getValue("threshold")
    minLineLength = tb.getValue("minLineLength")
    maxLineGap = tb.getValue("maxLineGap")

    return cv.HoughLinesP(canny, rho, np.pi/180, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)


def drawHoughLinesOverlay(frame, lines):
    if frame.ndim == 2:
        frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
    else:
        frame = frame.copy()

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                color = (np.random.randint(256), np.random.randint(256), np.random.randint(256))
                cv.line(frame, (x1, y1), (x2, y2), color, 1, cv.LINE_AA)
    
    return frame


def calcHoughLineMedian(lines, angle, delta=10):
    angle = (180 - angle) % 180
    if lines is not None:
        lines = lines.astype(np.float32)

        _x = lines[:,:,0] - lines[:,:,2]
        _y = lines[:,:,1] - lines[:,:,3]

        degree = np.arctan2(_y, _x) / np.pi * 180
        degree = (degree + 360) % 180

        print(degree)
        mask = np.bitwise_and((angle-delta)<=degree, degree<=(angle+delta))
        
        if np.any(mask):
            # median_x1, median_x2 = np.median(lines[mask])[::2]
            # return np.median(lines[mask], axis=0)
            return np.mean(lines[mask], axis=0)
            """weights = np.sqrt((_x[mask] - _y[mask])**2)
            if np.sum(weights):
                return np.average(lines[mask], weights=weights, axis=0)
            """
    
    return None, None, None, None


def sliding_window(frame, win_width, scan_shape, threshold=None, num_of_windows=48, step_window=5, tb=None):
    scan_height, scan_width = scan_shape
    height, width = frame.shape[:2]
    mid = width // 2

    win_height = height // num_of_windows
    win_half_width = win_width // 2

    if threshold is None:
        threshold = win_width * win_height // 4
    else:
        threshold = win_width * win_height * threshold

    scan_width = win_width if scan_width < win_width else scan_width
    scan_height = win_height if scan_height < win_height else scan_height

    scan_half_width = scan_width // 2

    left_histogram = np.sum(frame[-scan_height:, :mid], axis=0)
    right_histogram = np.sum(frame[-scan_height:, mid:], axis=0)
    
    left_pts = np.argwhere(left_histogram > np.max(left_histogram)*0.8).flatten()
    left_pts_group = np.split(left_pts, np.where(np.diff(left_pts) > step_window)[0]+1)
    left_centers = [ np.mean(pts, dtype=np.int32) for pts in left_pts_group if pts.size != 0 ]

    right_pts = np.argwhere(right_histogram > np.max(right_histogram)*0.8).flatten()
    right_pts_group = np.split(right_pts, np.where(np.diff(right_pts) > step_window)[0]+1)
    right_centers = [ mid + np.mean(pts, dtype=np.int32) for pts in right_pts_group if pts.size != 0 ]

    lx_group, ly_group, rx_group, ry_group = [], [], [], []

    nonzero_frame = frame.nonzero()
    nonzero_frame_y, nonzero_frame_x = nonzero_frame[0], nonzero_frame[1]

    """viewer"""
    viewer = None
    if tb is not None and tb.debug:
        viewer = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

    for left_center in left_centers:
        lx, ly = [], []

        for window_idx in range(num_of_windows):
            win_y1 = height - win_height*(window_idx+1)
            win_y2 = height - win_height*window_idx

            win_lx1 = left_center - win_half_width
            win_lx2 = left_center + win_half_width

            lx.append(left_center)
            ly.append((win_y1 + win_y2)/2)

            left_nonzero_indices = ((win_y1 <= nonzero_frame_y) & (nonzero_frame_y < win_y2)      \
                                    & (win_lx1 <= nonzero_frame_x) & (nonzero_frame_x < win_lx2)).nonzero()[0]
            

            if len(left_nonzero_indices) > threshold:
                scan_center = win_lx1 + win_half_width
                scan_x = scan_center - scan_half_width
                scan_y = win_y1 - scan_height

                next_histogram = np.sum(frame[scan_y:scan_y+scan_height, scan_x:scan_x+scan_width], axis=0)
                if next_histogram.size > 0:
                    first_idx = np.argmax(next_histogram)
                    last_idx = next_histogram.size - 1 - np.argmax(next_histogram[::-1])
                    left_center = scan_x + (first_idx + last_idx) // 2

                """viewer"""
                if tb is not None and tb.debug:
                    cv.rectangle(viewer, (win_lx1, win_y1), (win_lx2, win_y2), (0, 255, 0), 1) 
                    viewer[nonzero_frame_y[left_nonzero_indices], nonzero_frame_x[left_nonzero_indices]] = (0, 255, 0)
                    cv.rectangle(viewer, (scan_x, scan_y), (scan_x+scan_width, scan_y+scan_height), (255, 0, 0), 2)
            else:
                """viewer"""
                if tb is not None and tb.debug:
                    cv.rectangle(viewer, (win_lx1, win_y1), (win_lx2, win_y2), (0, 0, 255), 1) 
                    viewer[nonzero_frame_y[left_nonzero_indices], nonzero_frame_x[left_nonzero_indices]] = (0, 0, 255)
                break
                
        lx_group.append(lx)
        ly_group.append(ly)



    for right_center in right_centers:
        rx, ry = [], []

        for window_idx in range(num_of_windows):
            win_y1 = height - win_height*(window_idx+1)
            win_y2 = height - win_height*window_idx

            win_rx1 = right_center - win_half_width
            win_rx2 = right_center + win_half_width

            rx.append(right_center)
            ry.append((win_y1 + win_y2)/2)

            right_nonzero_indices = ((win_y1 <= nonzero_frame_y) & (nonzero_frame_y < win_y2)     \
                                    & (win_rx1 <= nonzero_frame_x) & (nonzero_frame_x < win_rx2)).nonzero()[0]

            
            if len(right_nonzero_indices) > threshold:
                scan_center = win_rx1 + win_half_width
                scan_x = scan_center - scan_half_width
                scan_y = win_y1 - scan_height

                next_histogram = np.sum(frame[scan_y:scan_y+scan_height, scan_x:scan_x+scan_width], axis=0)
                if next_histogram.size > 0:
                    first_idx = np.argmax(next_histogram)
                    last_idx = next_histogram.size - 1 - np.argmax(next_histogram[::-1])
                    right_center = scan_x + (first_idx + last_idx) // 2
                
                """viewer"""
                if tb is not None and tb.debug:
                    cv.rectangle(viewer, (win_rx1, win_y1), (win_rx2, win_y2), (0, 255, 0), 1) 
                    viewer[nonzero_frame_y[right_nonzero_indices], nonzero_frame_x[right_nonzero_indices]] = (0, 255, 0)
                    cv.rectangle(viewer, (scan_x, scan_y), (scan_x+scan_width, scan_y+scan_height), (255, 0, 0), 2)
            else:
                """viewer"""
                if tb is not None and tb.debug:
                    cv.rectangle(viewer, (win_rx1, win_y1), (win_rx2, win_y2), (0, 0, 255),1) 
                    viewer[nonzero_frame_y[right_nonzero_indices], nonzero_frame_x[right_nonzero_indices]] = (0, 0, 255)
                break
                

        rx_group.append(rx)
        ry_group.append(ry)
 

    left_coeff = [ np.polyfit(ly, lx, 2) for ly, lx in zip(ly_group, lx_group) if len(ly) > 3 and len(lx) > 3 ]
    right_coeff = [ np.polyfit(ry, rx, 2) for ry, rx in zip(ry_group, rx_group) if len(ry) > 3 and len(rx) > 3 ]
    
    return left_coeff, right_coeff, viewer


def get_radian_with_sliding(judge_y, roi_shape, left_coeff=None, right_coeff=None, history_x=None, image=None):
    height, width = roi_shape
    baseline_x, baseline_y = width//2, height
    left_x, right_x = 0, width

    if history_x is not None:
        left_x, right_x = history_x
    
    if left_coeff is not None:
        left_x = left_coeff[0]*judge_y**2 + left_coeff[1]*judge_y + left_coeff[2]
        if right_coeff is None:
            right_x = left_x + (left_x+right_x)

    if right_coeff is not None:
        right_x = right_coeff[0]*judge_y**2 + right_coeff[1]*judge_y + right_coeff[2]
        if left_coeff is None:
            left_x = right_x - (left_x+right_x)

    center_x = (left_x + right_x) // 2

    if image is not None:
        test = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        cv.line(test, (int(baseline_x), int(baseline_y)), (int(left_x), int(judge_y)), (0, 0, 255), 3)
        cv.line(test, (int(baseline_x), int(baseline_y)), (int(center_x), int(judge_y)), (0, 255, 0), 3)
        cv.line(test, (int(baseline_x), int(baseline_y)), (int(right_x), int(judge_y)), (255, 0, 0), 3)
        cv.imshow("judge_y", test)

    radian = np.arctan2(-judge_y+baseline_y, center_x-baseline_x)
    degree = np.degrees(radian)
    history_x = (left_x, right_x)

    return degree, history_x, image
    

def removeContours(frame, threshold_area, tb=None):
    if frame.ndim == 3:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    frame, contours, hierarchy = cv.findContours(frame, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    for i in range(len(contours)):
        if cv.contourArea(contours[i]) < threshold_area:
            cv.drawContours(frame, contours, i, 0, -1)

    return frame

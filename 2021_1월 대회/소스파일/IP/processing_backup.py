def thresholdHSV(frame, tb):
    hls = cv.cvtColor(frame, cv.COLOR_BGR2HLS)

    light_min = tb.getValue("light_min")
    light_max = tb.getValue("light_max")

    inRange = cv.inRange(hls, (0, light_min, 0), (255, light_max, 255))
    
    return cv.threshold(inRange, -1, 255, cv.THRESH_OTSU)


def threshold_HLS_L(frame, tb):
    hls = cv.cvtColor(frame, cv.COLOR_BGR2HLS)
    _, L, _ = cv.split(hls)
    blur = cv.GaussianBlur(L,(5, 5), 0)

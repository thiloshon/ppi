import http

import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import urllib.request
import numpy as np
import requests
from cvlib.object_detection import draw_bbox
import concurrent.futures

url = 'http://192.168.4.1:81/stream'


def run1():
    #cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)

    stream = urllib.request.urlopen(url)
    bytes = b''
    timer = 0

    while True:
        bytes += stream.read(1024)
        a = bytes.find(b'\xff\xd8')
        b = bytes.find(b'\xff\xd9')
        if a != -1 and b != -1:
            timer += 1
            jpg = bytes[a:b + 2]
            bytes = bytes[b + 2:]
            if len(jpg) == 0:
                continue
            i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), -1)
            i = cv2.flip(i, 0)

            if timer % 100 == 0:
                bbox, label, conf = cv.detect_common_objects(i)
                print(label)
                i = draw_bbox(i, bbox, label, conf)

                cv2.imwrite("capture" + str(timer) + ".jpg", i)

                if "cell phone" in label:
                    requests.get("http://192.168.4.1/control?var=car&val=1")

            #cv2.imshow('live transmission', i)

    #cv2.destroyAllWindows()


def run2():
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)
    while True:
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        im = cv2.imdecode(imgnp, -1)

        bbox, label, conf = cv.detect_common_objects(im)
        print(label)
        im = draw_bbox(im, bbox, label, conf)

        cv2.imshow('detection', im)
        key = cv2.waitKey(5)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    run1()
    # with concurrent.futures.ProcessPoolExecutor() as executer:
    #     f1 = executer.submit(run1)
        # f2 = executer.submit(run2)

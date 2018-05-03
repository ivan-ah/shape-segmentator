import glob
import os
import subprocess
import shutil

from PIL import Image
import numpy as np
import cv2

from simpleocr.files import open_image
from simpleocr.segmentation import ContourSegmenter, RawContourSegmenter
from simpleocr.feature_extraction import SimpleFeatureExtractor
from simpleocr.classification import KNNClassifier
from simpleocr.ocr import OCR, accuracy, show_differences


IMAGES_PATH = "/opt/projects/lettering/plantillas_lettering/"
DST_PATH = "/opt/projects/lettering/output/"
MIN_PIXELS = 200

def remove_background(src, dst):
    command = "/usr/local/bin/convert {} -fuzz 20% -transparent white -normalize {}".format(src, dst)
    subprocess.call(command, shell=True)


def convert_to_black_pixels(src, dst):
    command = "/usr/local/bin/convert {} -alpha extract -threshold 5% -negate -transparent white {}".format(src, dst)
    subprocess.call(command, shell=True)


def remove_transparency_and_saturate(src, dst):
    print "Converting from '{}' to '{}'".format(src, dst)
    remove_background(src, dst)
    convert_to_black_pixels(dst, dst)


def detect_boundaries(src, dst=None):
    img = cv2.imread(src)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)    
    
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [0,0,255]

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extract_contours2(src):
    img = cv2.imread(src)
    # threshold image
    ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                    127, 255, cv2.THRESH_BINARY)
    # find contours and get the external one
    image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE)
     
    # with each contour, draw boundingRect in green
    # a minAreaRect in red and
    # a minEnclosingCircle in blue
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a green rectangle to visualize the bounding rect
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
     
        # get the min area rect
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        # draw a red 'nghien' rectangle
        cv2.drawContours(img, [box], 0, (0, 0, 255))
     
        # finally, get the min enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(c)
        # convert all values to int
        # center = (int(x), int(y))
        # radius = int(radius)
        # img = cv2.circle(img, center, radius, (255, 0, 0), 2)

    # print(len(contours))
    # cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
     
    cv2.imshow("contours", img)
    cv2.imwrite(src, img)

def extract_contours(src):
    name, _ = os.path.splitext(os.path.basename(src))
    folder_name = os.path.join(DST_PATH, name)
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)

    os.makedirs(folder_name)

    im = cv2.imread(src)
    _image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    image, contours, hierarchy = cv2.findContours(_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    idx = 0 

    for cnt in contours:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        roi = im[y: y+h, x: x+w]

        _dst = os.path.join(folder_name, "{}_{}.png".format(name, str(idx)))

        if w < MIN_PIXELS and h < MIN_PIXELS:
            continue

        cv2.imwrite(_dst, roi)
        remove_background(_dst, _dst)

def main():
    for _file in glob.glob("{}/*_processed.png".format(IMAGES_PATH)):
        os.remove(_file)

    for image in glob.glob("{}/*.png".format(IMAGES_PATH)):
        print "Processing image: {}".format(image)
        dirname = os.path.dirname(image)
        name, _ = os.path.splitext(os.path.basename(image))
        dst_image = os.path.join(dirname, "{}_processed.png".format(name))

        # Imagemagick process
        remove_transparency_and_saturate(image, dst_image)

        extract_contours(dst_image)
        # detect_boundaries(dst_image)
        
        # segmenter = ContourSegmenter(blur_y=5, blur_x=5, block_size=11, c=10)
        # extractor = SimpleFeatureExtractor(feature_size=10, stretch=False)
        # classifier = KNNClassifier()
        # ocr = OCR(segmenter, extractor, classifier)

        # ocr.train(open_image('digits1'))

        # test_image = open_image(image)
        # test_chars, test_classes, test_segments = ocr.ocr(test_image, show_steps=True)

        # print("accuracy:", accuracy(test_image.ground.classes, test_classes))
        # print("OCRed text:\n", test_chars)

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import os
import subprocess
import shutil
import sys

from PIL import Image
import numpy as np
import cv2
from keras.preprocessing import image
from keras.models import Model

from src.vgg19 import VGG19
from src.imagenet_utils import preprocess_input
from src.plot_utils import plot_query_answer
from src.sort_utils import find_topk_unique
from src.kNN import kNN
from src.tSNE import plot_tsne

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

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [0, 0, 255]

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
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # get the min area rect
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        # draw a red 'nghien' rectangle
        cv2.drawContours(img, [box], 0, (0, 0, 255))

        # finally, get the min enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(c)

    cv2.imshow("contours", img)
    cv2.imwrite(src, img)


def is_contained(a, b):
    if a["min_x"] >= b["min_x"] and a["min_y"] >= b["min_y"] and a["max_x"] <= b["max_x"] and a["max_y"] <= b["max_y"]:
        return True
    return False


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

    rois_index = {}

    for cnt in contours:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        # roi = im[y: y + h, x: x + w]

        # TODO: if roi inside another roi, should pass

        if w < MIN_PIXELS and h < MIN_PIXELS:
            continue

        rois_index[idx] = {
            "min_x": x,
            "min_y": y,
            "max_x": x + w,
            "max_y": y + h
        }

    rois_values = rois_index.values()
    rois_elements = []
    for i in range(len(rois_values)):
        is_subset = False
        for j in range(i + 1, len(rois_values)):
            if is_contained(rois_values[i], rois_values[j]):
                is_subset = True
                break

        if is_subset is False:
            rois_elements.append(rois_values[i])

    for idx, values in rois_index.iteritems():
        _dst = os.path.join(folder_name, "{}_{}.png".format(name, str(idx)))
        roi = im[values["min_y"]: values["max_y"], values["min_x"]: values["max_x"]]
        cv2.imwrite(_dst, roi)
        remove_background(_dst, _dst)


def classifier(src_path):
    # ================================================
    # Load pre-trained model and remove higher level layers
    # ================================================
    print("Loading VGG19 pre-trained model...")
    base_model = VGG19(weights='imagenet')
    model = Model(input=base_model.input,
                  output=base_model.get_layer('block4_pool').output)

    # ================================================
    # Read images and convert them to feature vectors
    # ================================================
    imgs, filename_heads, X = [], [], []
    path = "db"
    print("Reading images from '{}' directory...\n".format(src_path))
    _files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(src_path) for f in filenames if
              os.path.splitext(f)[1] == '.png']
    print _files
    for f in _files:

        # Process filename
        filename = os.path.splitext(f)  # filename in directory
        filename_full = os.path.join(path, f)  # full path filename
        head, ext = filename[0], filename[1]
        if ext.lower() not in [".jpg", ".jpeg"]:
            continue

        # Read image file
        img = image.load_img(filename_full, target_size=(224, 224))  # load
        imgs.append(np.array(img))  # image
        filename_heads.append(head)  # filename head

        # Pre-process for model input
        img = image.img_to_array(img)  # convert to array
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = model.predict(img).flatten()  # features
        X.append(features)  # append feature extractor

    X = np.array(X)  # feature vectors
    imgs = np.array(imgs)  # images
    print("imgs.shape = {}".format(imgs.shape))
    print("X_features.shape = {}\n".format(X.shape))

    # ===========================
    # Find k-nearest images to each image
    # ===========================
    n_neighbours = 5 + 1  # +1 as itself is most similar
    knn = kNN()  # kNN model
    knn.compile(n_neighbors=n_neighbours, algorithm="brute", metric="cosine")
    knn.fit(X)

    # ==================================================
    # Plot recommendations for each image in database
    # ==================================================
    output_rec_dir = os.path.join("output", "rec")
    if not os.path.exists(output_rec_dir):
        os.makedirs(output_rec_dir)
    n_imgs = len(imgs)
    ypixels, xpixels = imgs[0].shape[0], imgs[0].shape[1]
    for ind_query in range(n_imgs):
        # Find top-k closest image feature vectors to each vector
        print("[{}/{}] Plotting similar image recommendations for: {}".format(ind_query + 1, n_imgs,
                                                                              filename_heads[ind_query]))
        distances, indices = knn.predict(np.array([X[ind_query]]))
        distances = distances.flatten()
        indices = indices.flatten()
        indices, distances = find_topk_unique(indices, distances, n_neighbours)

        # Plot recommendations
        rec_filename = os.path.join(output_rec_dir, "{}_rec.png".format(filename_heads[ind_query]))
        x_query_plot = imgs[ind_query].reshape((-1, ypixels, xpixels, 3))
        x_answer_plot = imgs[indices].reshape((-1, ypixels, xpixels, 3))
        plot_query_answer(x_query=x_query_plot,
                          x_answer=x_answer_plot[1:],  # remove itself
                          filename=rec_filename)

    # ===========================
    # Plot tSNE
    # ===========================
    output_tsne_dir = os.path.join("output")
    if not os.path.exists(output_tsne_dir):
        os.makedirs(output_tsne_dir)
    tsne_filename = os.path.join(output_tsne_dir, "tsne.png")
    print("Plotting tSNE to {}...".format(tsne_filename))
    plot_tsne(imgs, X, tsne_filename)


def main():
    for _file in glob.glob("{}/*.jpg".format(IMAGES_PATH)):
        command = "./bg_removal {} {} {} {} {}".format(_file, "white", 50, 50, _file.replace(".jpg", ".png"))
        subprocess.call(command, shell=True)
        os.remove(_file)

    for _file in glob.glob("{}/*_processed.png".format(IMAGES_PATH)):
        os.remove(_file)

    for image in glob.glob("{}/*.png".format(IMAGES_PATH)):
        print "Processing image: {}".format(image)
        dirname = os.path.dirname(image)
        name, _ = os.path.splitext(os.path.basename(image))
        dst_image = os.path.join(dirname, "{}_processed.png".format(name))

        # Imagemagick process
        remove_transparency_and_saturate(image, dst_image) ## !!!!

        extract_contours(dst_image)

        # classifier(DST_PATH)

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

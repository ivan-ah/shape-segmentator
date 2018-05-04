#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

DST_PATH = "/opt/projects/lettering/letters/"
ABC = "abcdefghijklmnopqrstuvwxyz"
NUMBERS = "0123456789"


def create_folders(name):
    path = os.path.join(DST_PATH, name)
    if not os.path.exists(path):
        os.makedirs(path)


for letter in ABC:
    create_folders(letter)

for letter in ABC.upper():
    create_folders("_" + letter)

for number in NUMBERS:
    create_folders(number)

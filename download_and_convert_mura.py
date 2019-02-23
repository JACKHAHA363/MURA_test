# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import re
import os
from os import getcwd
from os.path import exists, isdir, isfile, join
import shutil
import numpy as np
import pandas as pd


class ImageString(object):
    _patient_re = re.compile(r'patient(\d+)')
    _study_re = re.compile(r'study(\d+)')
    _image_re = re.compile(r'image(\d+)')
    _study_type_re = re.compile(r'XR_(\w+)')

    def __init__(self, img_filename):
        self.img_filename = img_filename
        self.patient = self._parse_patient()
        self.study = self._parse_study()
        self.image_num = self._parse_image()
        self.study_type = self._parse_study_type()
        self.image = self._parse_image()
        self.normal = self._parse_normal()

    def flat_file_name(self):
        return "{}_{}_patient{}_study{}_image{}.png".format(self.normal, self.study_type, self.patient, self.study,
                                                            self.image, self.normal)

    def _parse_patient(self):
        return int(self._patient_re.search(self.img_filename).group(1))

    def _parse_study(self):
        return int(self._study_re.search(self.img_filename).group(1))

    def _parse_image(self):
        return int(self._image_re.search(self.img_filename).group(1))

    def _parse_study_type(self):
        return self._study_type_re.search(self.img_filename).group(1)

    def _parse_normal(self):
        return "normal" if ("negative" in self.img_filename) else "abnormal"


def process(label_csv, proc_dir):
    labels = pd.read_csv(label_csv)
    for img_folder, label in labels.itertuples(index=False):
        assert ("negative" in img_folder) is (label is 0)
        for img in os.listdir(img_folder):
            enc = ImageString(join(img_folder, img))
            cat_dir = join(proc_dir, enc.normal)
            if not os.path.exists(cat_dir):
                os.makedirs(cat_dir)
            shutil.copy2(enc.img_filename, join(cat_dir, enc.flat_file_name()))


if __name__ == '__main__':
    # data
    # ├── train
    # │   ├── abnormal
    # │   └── normal
    # └── val
    #     ├── abnormal
    #     └── normal
    proj_folder = os.path.dirname(__file__)

    # Data loading code
    orig_data_dir = join(getcwd(), 'MURA-v1.1')
    val_dir = join(orig_data_dir, 'valid')
    val_img_csv = join(orig_data_dir, 'valid_image_paths.csv')
    val_label_csv = join(orig_data_dir, 'valid_labeled_studies.csv')
    assert isdir(orig_data_dir) and isdir(val_dir)
    assert exists(val_img_csv) and isfile(val_img_csv) and exists(val_label_csv) and isfile(val_label_csv)

    process(label_csv=val_label_csv,
            proc_dir=join(proj_folder, 'data', 'val'))

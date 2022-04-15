import pandas as pd
import numpy as np
import os
import cv2

from matplotlib import pyplot as plt

from lib.visualization import plotting
from lib.visualization.video import play_trip

from tqdm import tqdm

class VisualOdometry:

    def __init__(self, data_directory) -> None:
        self.k, self.p = self._load_calib(os.path.join(data_directory, "calib.txt"))
        self.poses = self._load_poses(os.path.join(data_directory, 'poses.txt'))
        self.images = self._load_images(os.path.join(data_directory, "image_l "))
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX__LSH = 6
        index_param = dict(algorithm=FLANN_INDEX__LSH, table_number=6, ket_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_param=index_param, search_params=search_params)

    @staticmethod
    def _load_calib(filepath):
        """
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep = ' ')
            p = np.reshape(params, (3, 4))
            k = p[:3, :3]
        return p, k

    @staticmethod
    def _load_poses(filepath):
        """
        """
        poses = list()

        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep = " ")
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):
        """
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    @staticmethod
    def _from_transf(R, T):
        """
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = T
        return T

    def get_matches(self, i):
        """
        """
        keypoints1, descriptors1 = self.orb.detectAndCompute(self.images[i - 1], None)
        keypoints2, descriptors2 = self.orb.detectAndCompute(self.images[i], None)

        matches = self.flann.KnnMatch(descriptors1, descriptors2, k = 2)

        good = list()
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append(m)
        
        q1 = np.float32([keypoints1[m.queryIdx].t for m in good])
        q2 = np.float32([keypoints2[m.queryIdx].t for m in good])

        draw_params = dict(matchColor)
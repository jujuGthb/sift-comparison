"""
SIFT Comparison Executor: Compares two images using pre-computed SIFT descriptors.
SIFT feature extraction is handled by an external SIFT block.
Visualization is handled by an external Draw Keypoint block.
Runs entirely locally using OpenCV — no external API required.
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.SIFTComparison.src.utils.response import build_response_sift_comparison
from components.SIFTComparison.src.models.PackageModel import PackageModel


class SIFTComparison(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))

        self.good_matches_threshold = self.request.get_param("GoodMatchesThreshold")
        self.ratio_threshold = self.request.get_param("RatioThreshold")
        self.matcher = self.request.get_param("Matcher")
        self.descriptors_input_1 = self.request.get_param("InputDescriptors1")
        self.descriptors_input_2 = self.request.get_param("InputDescriptors2")
        self.visualization_input_1 = self.request.get_param("InputVisualization1")
        self.visualization_input_2 = self.request.get_param("InputVisualization2")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def run(self):
        try:

            descriptors1 = np.array(self.descriptors_input_1, dtype=np.float32)
            descriptors2 = np.array(self.descriptors_input_2, dtype=np.float32)

            print(f"[SIFT] descriptors1 shape: {descriptors1.shape}")
            print(f"[SIFT] descriptors2 shape: {descriptors2.shape}")

            if descriptors1 is None or descriptors2 is None or \
               len(descriptors1) < 2 or len(descriptors2) < 2:
                print(f"[SIFT] Early return — not enough descriptors")
                self.images_match = False
                self.good_matches_count = 0
                self.keypoints1 = []
                self.keypoints2 = []
                self.descriptors1 = self.descriptors_input_1
                self.descriptors2 = self.descriptors_input_2
                self.visualization1 = self.visualization_input_1
                self.visualization2 = self.visualization_input_2
                self.visualization_matches = None
                return build_response_sift_comparison(context=self)

            if self.matcher == "BFMatcher":
                matcher = cv2.BFMatcher(cv2.NORM_L2)
            else:
                matcher = cv2.FlannBasedMatcher(
                    dict(algorithm=1, trees=5),
                    dict(checks=50)
                )

            matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

            if len(matches) > 0:
                m, n = matches[0]
                print(f"[SIFT] sample ratio: m.distance={m.distance:.4f}, n.distance={n.distance:.4f}, ratio={m.distance/n.distance:.4f}")

            good_matches = []
            for m, n in matches:
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append([m])

            self.good_matches_count = len(good_matches)
            self.images_match = self.good_matches_count >= self.good_matches_threshold

            print(f"[SIFT] good_matches_count: {self.good_matches_count}")
            print(f"[SIFT] images_match: {self.images_match}")

            keypoints1_indices = [m[0].queryIdx for m in good_matches]
            keypoints2_indices = [m[0].trainIdx for m in good_matches]

            self.keypoints1 = keypoints1_indices
            self.keypoints2 = keypoints2_indices

            self.descriptors1 = self.descriptors_input_1
            self.descriptors2 = self.descriptors_input_2
            self.visualization1 = self.visualization_input_1
            self.visualization2 = self.visualization_input_2
            self.visualization_matches = None

        except Exception as e:
            self.images_match = False
            self.good_matches_count = 0
            self.keypoints1 = []
            self.keypoints2 = []
            self.descriptors1 = self.descriptors_input_1 if hasattr(self, 'descriptors_input_1') else None
            self.descriptors2 = self.descriptors_input_2 if hasattr(self, 'descriptors_input_2') else None
            self.visualization1 = self.visualization_input_1 if hasattr(self, 'visualization_input_1') else None
            self.visualization2 = self.visualization_input_2 if hasattr(self, 'visualization_input_2') else None
            self.visualization_matches = None
            print(f"[SIFTComparison] Error: {str(e)}")

        return build_response_sift_comparison(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
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
        self.sift_output_1 = self.request.get_param("InputSIFTOutput1")
        self.sift_output_2 = self.request.get_param("InputSIFTOutput2")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _extract_keypoints_and_descriptors(self, sift_output):
        keypoints_dicts = []
        descriptors = []
        for detection in sift_output:
            for kp in detection.get("keyPoints", []):
                keypoints_dicts.append({
                    "pt": (kp["cx"], kp["cy"]),
                    "size": kp["size"],
                    "angle": kp["angle"],
                    "response": kp["response"],
                    "octave": kp["octave"],
                })
                descriptors.append(kp["descriptor"])
        return keypoints_dicts, np.array(descriptors, dtype=np.float32)

    def run(self):
        try:
            keypoints1_dicts, descriptors1 = self._extract_keypoints_and_descriptors(self.sift_output_1)
            keypoints2_dicts, descriptors2 = self._extract_keypoints_and_descriptors(self.sift_output_2)

            print(f"[SIFT] keypoints1 count: {len(keypoints1_dicts)}")
            print(f"[SIFT] keypoints2 count: {len(keypoints2_dicts)}")
            print(f"[SIFT] descriptors1 shape: {descriptors1.shape}")
            print(f"[SIFT] descriptors2 shape: {descriptors2.shape}")

            if len(descriptors1) < 2 or len(descriptors2) < 2:
                print(f"[SIFT] Early return — not enough descriptors")
                self.images_match = False
                self.good_matches_count = 0
                self.keypoints1 = keypoints1_dicts
                self.keypoints2 = keypoints2_dicts
                self.descriptors1 = descriptors1.tolist()
                self.descriptors2 = descriptors2.tolist()
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
            self.keypoints1 = keypoints1_dicts
            self.keypoints2 = keypoints2_dicts
            self.descriptors1 = descriptors1.tolist()
            self.descriptors2 = descriptors2.tolist()

            print(f"[SIFT] good_matches_count: {self.good_matches_count}")
            print(f"[SIFT] images_match: {self.images_match}")

        except Exception as e:
            self.images_match = False
            self.good_matches_count = 0
            self.keypoints1 = []
            self.keypoints2 = []
            self.descriptors1 = None
            self.descriptors2 = None
            print(f"[SIFTComparison] Error: {str(e)}")

        return build_response_sift_comparison(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
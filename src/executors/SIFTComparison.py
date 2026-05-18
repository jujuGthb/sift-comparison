"""
SIFT Comparison Executor: Compares two images using SIFT feature matching.
Runs entirely locally using OpenCV — no external API required.
"""

import os
import sys
import base64
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
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
        self.visualize = self.request.get_param("Visualize")
        self.image_selector_1 = self.request.get_param("InputImage1")
        self.image_selector_2 = self.request.get_param("InputImage2")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _apply_sift(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)

        visualization = None
        if self.visualize:
            visualization = cv2.drawKeypoints(gray, kp, None)

        keypoints_dicts = [
            {
                "pt": (point.pt[0], point.pt[1]),
                "size": point.size,
                "angle": point.angle,
                "response": point.response,
                "octave": point.octave,
                "class_id": point.class_id,
            }
            for point in kp
        ]

        return visualization, kp, keypoints_dicts, des

    def run(self):
        try:
            img1 = Image.get_frame(img=self.image_selector_1, redis_db=self.redis_db)
            img2 = Image.get_frame(img=self.image_selector_2, redis_db=self.redis_db)

            image1 = img1.value
            image2 = img2.value

            _, buf1 = cv2.imencode('.jpg', image1)
            image1 = cv2.imdecode(buf1, cv2.IMREAD_COLOR)

            _, buf2 = cv2.imencode('.jpg', image2)
            image2 = cv2.imdecode(buf2, cv2.IMREAD_COLOR)

            print(f"[SIFT] image1 type after decode: {type(image1)}")
            print(f"[SIFT] image1 dtype after decode: {image1.dtype}")
            print(f"[SIFT] image2 type after decode: {type(image2)}")
            print(f"[SIFT] image2 dtype after decode: {image2.dtype}")

            viz1, kp1, keypoints1, descriptors1 = self._apply_sift(image1)
            viz2, kp2, keypoints2, descriptors2 = self._apply_sift(image2)

            print(f"[SIFT] keypoints1 count: {len(keypoints1)}")
            print(f"[SIFT] keypoints2 count: {len(keypoints2)}")
            print(f"[SIFT] descriptors1 shape: {descriptors1.shape if descriptors1 is not None else 'None'}")
            print(f"[SIFT] descriptors2 shape: {descriptors2.shape if descriptors2 is not None else 'None'}")

            if descriptors1 is None or descriptors2 is None or \
               len(descriptors1) < 2 or len(descriptors2) < 2:
                print(f"[SIFT] Early return — not enough descriptors")
                self.images_match = False
                self.good_matches_count = 0
                self.keypoints1 = keypoints1
                self.keypoints2 = keypoints2
                self.descriptors1 = descriptors1.tolist() if descriptors1 is not None else None
                self.descriptors2 = descriptors2.tolist() if descriptors2 is not None else None
                self.visualization1 = None
                self.visualization2 = None
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

            good_matches = []
            for m, n in matches:
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append([m])

            self.good_matches_count = len(good_matches)
            self.images_match = self.good_matches_count >= self.good_matches_threshold

            print(f"[SIFT] good_matches_count: {self.good_matches_count}")
            print(f"[SIFT] images_match: {self.images_match}")

            self.keypoints1 = keypoints1
            self.keypoints2 = keypoints2
            self.descriptors1 = descriptors1.tolist() if descriptors1 is not None else None
            self.descriptors2 = descriptors2.tolist() if descriptors2 is not None else None

            self.visualization1 = None
            self.visualization2 = None
            self.visualization_matches = None

            if self.visualize:
                if viz1 is not None:
                    _, buf1 = cv2.imencode('.jpg', viz1)
                    self.visualization1 = base64.b64encode(buf1).decode('utf-8')

                if viz2 is not None:
                    _, buf2 = cv2.imencode('.jpg', viz2)
                    self.visualization2 = base64.b64encode(buf2).decode('utf-8')

                if kp1 and kp2:
                    if self.matcher == "BFMatcher":
                        draw_params = dict(
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                        )
                    else:
                        draw_params = dict(
                            matchColor=(0, 255, 0),
                            singlePointColor=(0, 0, 255),
                            flags=cv2.DrawMatchesFlags_DEFAULT,
                        )
                    viz_matches = cv2.drawMatchesKnn(
                        image1, kp1,
                        image2, kp2,
                        good_matches,
                        None,
                        **draw_params,
                    )
                    _, buf_m = cv2.imencode('.jpg', viz_matches)
                    self.visualization_matches = base64.b64encode(buf_m).decode('utf-8')

        except Exception as e:
            self.images_match = False
            self.good_matches_count = 0
            self.keypoints1 = []
            self.keypoints2 = []
            self.descriptors1 = None
            self.descriptors2 = None
            self.visualization1 = None
            self.visualization2 = None
            self.visualization_matches = None
            print(f"[SIFTComparison] Error: {str(e)}")

        return build_response_sift_comparison(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
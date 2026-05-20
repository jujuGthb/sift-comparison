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

    def _keypoints_to_cv2(self, keypoints_dicts):
        keypoints = []
        for kp in keypoints_dicts:
            pt = kp.get("pt", (0, 0))
            keypoints.append(cv2.KeyPoint(
                x=float(pt[0]),
                y=float(pt[1]),
                size=float(kp.get("size", 1)),
                angle=float(kp.get("angle", -1)),
                response=float(kp.get("response", 0)),
                octave=int(kp.get("octave", 0)),
                class_id=-1,
            ))
        return keypoints

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

            self.visualization1 = None
            self.visualization2 = None
            self.visualization_matches = None

            if self.visualize:
                img1 = Image.get_frame(img=self.image_selector_1, redis_db=self.redis_db)
                img2 = Image.get_frame(img=self.image_selector_2, redis_db=self.redis_db)

                _, buf1 = cv2.imencode('.jpg', img1.value)
                image1 = cv2.imdecode(buf1, cv2.IMREAD_COLOR)

                _, buf2 = cv2.imencode('.jpg', img2.value)
                image2 = cv2.imdecode(buf2, cv2.IMREAD_COLOR)

                kp1 = self._keypoints_to_cv2(keypoints1_dicts)
                kp2 = self._keypoints_to_cv2(keypoints2_dicts)

                gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

                viz1 = cv2.drawKeypoints(gray1, kp1, None)
                viz2 = cv2.drawKeypoints(gray2, kp2, None)

                _, buf_v1 = cv2.imencode('.jpg', viz1)
                self.visualization1 = base64.b64encode(buf_v1).decode('utf-8')

                _, buf_v2 = cv2.imencode('.jpg', viz2)
                self.visualization2 = base64.b64encode(buf_v2).decode('utf-8')

                if self.matcher == "BFMatcher":
                    draw_params = dict(flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
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
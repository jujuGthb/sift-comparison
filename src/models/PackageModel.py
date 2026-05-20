from pydantic import validator
from typing import List, Optional, Union, Literal, Any
from sdks.novavision.src.base.model import (
    Package, Image, Inputs, Outputs, Configs, Response, Request, Output, Input, Config
)


class InputDescriptors1(Input):
    name: Literal["InputDescriptors1"] = "InputDescriptors1"
    value: Optional[Any]
    type: Literal["object"] = "object"

    class Config:
        title = "Descriptors 1"


class InputDescriptors2(Input):
    name: Literal["InputDescriptors2"] = "InputDescriptors2"
    value: Optional[Any]
    type: Literal["object"] = "object"

    class Config:
        title = "Descriptors 2"


class InputVisualization1(Input):
    name: Literal["InputVisualization1"] = "InputVisualization1"
    value: Optional[Any]
    type: Literal["object"] = "object"

    class Config:
        title = "Visualization 1"


class InputVisualization2(Input):
    name: Literal["InputVisualization2"] = "InputVisualization2"
    value: Optional[Any]
    type: Literal["object"] = "object"

    class Config:
        title = "Visualization 2"


class OutputImagesMatch(Output):
    name: Literal["ImagesMatch"] = "ImagesMatch"
    value: Optional[bool]
    type: Literal["bool"] = "bool"

    class Config:
        title = "Images Match"


class OutputGoodMatchesCount(Output):
    name: Literal["GoodMatchesCount"] = "GoodMatchesCount"
    value: Optional[int]
    type: Literal["number"] = "number"

    class Config:
        title = "Good Matches Count"


class OutputKeypoints1(Output):
    name: Literal["Keypoints1"] = "Keypoints1"
    value: Optional[Any]
    type: Literal["object"] = "object"

    class Config:
        title = "Keypoints 1"


class OutputKeypoints2(Output):
    name: Literal["Keypoints2"] = "Keypoints2"
    value: Optional[Any]
    type: Literal["object"] = "object"

    class Config:
        title = "Keypoints 2"


class OutputDescriptors1(Output):
    name: Literal["Descriptors1"] = "Descriptors1"
    value: Optional[Any]
    type: Literal["object"] = "object"

    class Config:
        title = "Descriptors 1"


class OutputDescriptors2(Output):
    name: Literal["Descriptors2"] = "Descriptors2"
    value: Optional[Any]
    type: Literal["object"] = "object"

    class Config:
        title = "Descriptors 2"


class OutputVisualization1(Output):
    name: Literal["Visualization1"] = "Visualization1"
    value: Optional[Any]
    type: Literal["Image"] = "Image"

    class Config:
        title = "Visualization 1"


class OutputVisualization2(Output):
    name: Literal["Visualization2"] = "Visualization2"
    value: Optional[Any]
    type: Literal["Image"] = "Image"

    class Config:
        title = "Visualization 2"


class OutputVisualizationMatches(Output):
    name: Literal["VisualizationMatches"] = "VisualizationMatches"
    value: Optional[Any]
    type: Literal["Image"] = "Image"

    class Config:
        title = "Visualization Matches"


class GoodMatchesThreshold(Config):
    """
    Minimum number of good feature matches required to consider the two images as matching.
    Lower values (e.g. 20-30) are more lenient. Higher values (e.g. 80-100) are stricter.
    Default is 50.
    """
    name: Literal["GoodMatchesThreshold"] = "GoodMatchesThreshold"
    value: int = 50
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Good Matches Threshold"
        json_schema_extra = {"shortDescription": "Min matches to consider a match"}


class RatioThreshold(Config):
    """
    Ratio threshold for Lowe's ratio test (0.0-1.0).
    Lower values (e.g. 0.6) are stricter. Higher values (e.g. 0.8) are more lenient.
    Default is 0.7.
    """
    name: Literal["RatioThreshold"] = "RatioThreshold"
    value: float = 0.7
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Ratio Threshold"
        json_schema_extra = {"shortDescription": "Lowe's ratio test (0.0-1.0)"}


class MatcherFlann(Config):
    """
    FlannBasedMatcher uses FLANN for efficient approximate nearest neighbor search.
    Faster for large descriptor sets. Recommended for most use cases.
    """
    name: Literal["FlannBasedMatcher"] = "FlannBasedMatcher"
    value: Literal["FlannBasedMatcher"] = "FlannBasedMatcher"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "FLANN Based Matcher"
        json_schema_extra = {"shortDescription": "Fast, approximate matching"}


class MatcherBF(Config):
    """
    BFMatcher uses brute force matching with L2 norm.
    Exact matching but slower for large descriptor sets.
    """
    name: Literal["BFMatcher"] = "BFMatcher"
    value: Literal["BFMatcher"] = "BFMatcher"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Brute Force Matcher"
        json_schema_extra = {"shortDescription": "Exact, slower matching"}


class Matcher(Config):
    """
    Select the matcher algorithm to use for comparing SIFT descriptors.
    FlannBasedMatcher is faster. BFMatcher is more exact but slower.
    """
    name: Literal["Matcher"] = "Matcher"
    value: Union[MatcherFlann, MatcherBF]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Matcher Algorithm"
        json_schema_extra = {"shortDescription": "FLANN or Brute Force"}


class SIFTComparisonConfigs(Configs):
    GoodMatchesThreshold: GoodMatchesThreshold
    RatioThreshold: RatioThreshold
    Matcher: Matcher


class SIFTComparisonInputs(Inputs):
    InputDescriptors1: InputDescriptors1
    InputDescriptors2: InputDescriptors2
    InputVisualization1: InputVisualization1
    InputVisualization2: InputVisualization2


class SIFTComparisonOutputs(Outputs):
    ImagesMatch: OutputImagesMatch
    GoodMatchesCount: OutputGoodMatchesCount
    Keypoints1: OutputKeypoints1
    Keypoints2: OutputKeypoints2
    Descriptors1: OutputDescriptors1
    Descriptors2: OutputDescriptors2
    Visualization1: OutputVisualization1
    Visualization2: OutputVisualization2
    VisualizationMatches: OutputVisualizationMatches


class SIFTComparisonRequest(Request):
    inputs: Optional[SIFTComparisonInputs]
    configs: SIFTComparisonConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class SIFTComparisonResponse(Response):
    outputs: SIFTComparisonOutputs


class SIFTComparison(Config):
    """
    Compares two images using pre-computed SIFT descriptors from an external SIFT block.
    Accepts drawn visualizations from a Draw Keypoint block and passes them through as output.
    Applies Lowe's ratio test and outputs all 9 results including match count and boolean result.
    Keypoints are extracted internally from the matching results.
    """
    name: Literal["SIFTComparison"] = "SIFTComparison"
    value: Union[SIFTComparisonRequest, SIFTComparisonResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "SIFT Comparison"
        json_schema_extra = {"target": {"value": 0}, "shortDescription": "Feature-based image matching"}


class ConfigExecutor(Config):
    """
    SIFT Comparison — compares two images using pre-computed SIFT descriptors.
    No API key required. Runs entirely on the local machine using OpenCV.
    """
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[SIFTComparison]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {"target": "value"}


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    name: Literal["SIFTComparison"] = "SIFTComparison"
    configs: PackageConfigs
    type: Literal["component"] = "component"
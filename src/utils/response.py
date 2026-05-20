from sdks.novavision.src.helper.package import PackageHelper
from components.SIFTComparison.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    SIFTComparison,
    SIFTComparisonResponse,
    SIFTComparisonOutputs,
    OutputImagesMatch,
    OutputGoodMatchesCount,
    OutputKeypoints1,
    OutputKeypoints2,
    OutputDescriptors1,
    OutputDescriptors2,
    OutputVisualization1,
    OutputVisualization2,
    OutputVisualizationMatches,
)


def build_response_sift_comparison(context):
    images_match = OutputImagesMatch(value=context.images_match)
    good_matches_count = OutputGoodMatchesCount(value=context.good_matches_count)
    keypoints1 = OutputKeypoints1(value=context.keypoints1)
    keypoints2 = OutputKeypoints2(value=context.keypoints2)
    descriptors1 = OutputDescriptors1(value=context.descriptors1)
    descriptors2 = OutputDescriptors2(value=context.descriptors2)
    visualization1 = OutputVisualization1(value=context.visualization1)
    visualization2 = OutputVisualization2(value=context.visualization2)
    visualization_matches = OutputVisualizationMatches(value=context.visualization_matches)

    outputs = SIFTComparisonOutputs(
        ImagesMatch=images_match,
        GoodMatchesCount=good_matches_count,
        Keypoints1=keypoints1,
        Keypoints2=keypoints2,
        Descriptors1=descriptors1,
        Descriptors2=descriptors2,
        Visualization1=visualization1,
        Visualization2=visualization2,
        VisualizationMatches=visualization_matches,
    )

    response = SIFTComparisonResponse(outputs=outputs)
    executor = SIFTComparison(value=response)
    configExecutor = ConfigExecutor(value=executor)
    packageConfigs = PackageConfigs(executor=configExecutor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)

    return package.build_model(context)
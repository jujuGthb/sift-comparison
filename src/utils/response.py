from sdks.novavision.src.helper.package import PackageHelper
from components.SIFTComparison.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    SIFTComparison,
    SIFTComparisonResponse,
    SIFTComparisonOutputs,
    OutputDetectionResult,
    OutputVisualizationMatches,
)


def build_response_sift_comparison(context):
    detection_result = OutputDetectionResult(value=context.detection_result)
    visualization_matches = OutputVisualizationMatches(value=context.visualization_matches)

    outputs = SIFTComparisonOutputs(
        DetectionResult=detection_result,
        VisualizationMatches=visualization_matches,
    )

    response = SIFTComparisonResponse(outputs=outputs)
    executor = SIFTComparison(value=response)
    configExecutor = ConfigExecutor(value=executor)
    packageConfigs = PackageConfigs(executor=configExecutor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)

    return package.build_model(context)
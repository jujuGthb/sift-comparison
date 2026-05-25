from sdks.novavision.src.helper.package import PackageHelper
from components.SIFTComparison.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    SIFTComparison,
    SIFTComparisonResponse,
    SIFTComparisonOutputs,
    OutputDetectionResult,
    OutputDetections,
)


def build_response_sift_comparison(context):
    detection_result = OutputDetectionResult(value=context.detection_result)
    output_detections = OutputDetections(value=context.output_detections)

    outputs = SIFTComparisonOutputs(
        DetectionResult=detection_result,
        OutputDetections=output_detections,
    )

    response = SIFTComparisonResponse(outputs=outputs)
    executor = SIFTComparison(value=response)
    configExecutor = ConfigExecutor(value=executor)
    packageConfigs = PackageConfigs(executor=configExecutor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)

    return package.build_model(context)
package com.afriste.tn5000ortdemo

enum class OrtRuntimeProfile(
    val label: String,
    val intraOpThreads: Int? = null,
    val interOpThreads: Int? = null,
    val useXnnpack: Boolean = false,
    val useNnapi: Boolean = false,
    val nnapiUseFp16: Boolean = false,
    val nnapiUseNchw: Boolean = false,
) {
    CPU_DEFAULT(label = "CPU default"),
    CPU_1T(label = "CPU 1T", intraOpThreads = 1, interOpThreads = 1),
    CPU_2T(label = "CPU 2T", intraOpThreads = 2, interOpThreads = 1),
    CPU_4T(label = "CPU 4T", intraOpThreads = 4, interOpThreads = 1),
    CPU_8T(label = "CPU 8T", intraOpThreads = 8, interOpThreads = 1),
    XNNPACK_4T(label = "XNNPACK 4T", intraOpThreads = 4, useXnnpack = true),
    NNAPI(label = "NNAPI", useNnapi = true),
    NNAPI_FP16_NCHW(label = "NNAPI FP16 NCHW", useNnapi = true, nnapiUseFp16 = true, nnapiUseNchw = true),
}

enum class DeploymentMode(
    val title: String,
    val useHflipTta: Boolean,
    val assetNames: List<String>,
    val inputSize: Int = 256,
    val runtimeProfile: OrtRuntimeProfile = OrtRuntimeProfile.CPU_DEFAULT,
) {
    FAST(
        title = "Fast",
        useHflipTta = false,
        assetNames = listOf("tn5000_current_mainline.onnx"),
    ),
    ACCURATE(
        title = "Accurate",
        useHflipTta = true,
        assetNames = listOf("tn5000_current_mainline.onnx"),
    ),
    ENSEMBLE(
        title = "Ensemble",
        useHflipTta = true,
        assetNames = listOf(
            "tn5000_current_mainline.onnx",
            "tn5000_epoch060_bal_acc_0.9195.onnx",
            "tn5000_epoch054_bal_acc_0.9169.onnx",
        ),
    ),
    GGG_MCA(
        title = "GGG-MCA",
        useHflipTta = false,
        assetNames = listOf("tn5000_ggg_mca_s27_current.onnx"),
    ),
    GGG_MCA_TTA(
        title = "GGG-MCA+TTA",
        useHflipTta = true,
        assetNames = listOf("tn5000_ggg_mca_s27_current.onnx"),
    ),
    GGG_MCA_ENSEMBLE(
        title = "GGG-MCA Ensemble",
        useHflipTta = true,
        assetNames = listOf(
            "tn5000_ggg_mca_s27_current.onnx",
            "tn5000_ggg_mca_s27_epoch047.onnx",
            "tn5000_ggg_mca_s27_epoch046.onnx",
        ),
    ),
    BASELINE_MOBILENETV3(
        title = "MobileNetV3",
        useHflipTta = false,
        assetNames = listOf("tn5000_baseline_mobilenetv3_s27.onnx"),
        inputSize = 256,
    ),
    BASELINE_REPVIT_M11(
        title = "RepViT-M1.1",
        useHflipTta = false,
        assetNames = listOf("tn5000_baseline_repvit_m11_s27.onnx"),
        inputSize = 256,
    ),
    BASELINE_EFFICIENTFORMER_L1(
        title = "EfficientFormer-L1",
        useHflipTta = false,
        assetNames = listOf("tn5000_baseline_efficientformer_l1_s27.onnx"),
        inputSize = 224,
    ),
    BASELINE_CONVNEXT_TINY(
        title = "ConvNeXt-Tiny",
        useHflipTta = false,
        assetNames = listOf("tn5000_baseline_convnext_tiny_s27.onnx"),
        inputSize = 256,
    ),
    MOBILE_RECOVERY_EFFFORMER_L1_KD(
        title = "EffFormer-L1+KD",
        useHflipTta = false,
        assetNames = listOf("tn5000_mobile_recovery_effformerl1_kd_mobile_recovery_s17.onnx"),
        inputSize = 224,
    ),
    MOBILE_RECOVERY_EFFFORMER_L1_ECA_KD(
        title = "EffFormer-L1+ECA+KD",
        useHflipTta = false,
        assetNames = listOf("tn5000_mobile_recovery_effformerl1_eca_kd_mobile_recovery_s17.onnx"),
        inputSize = 224,
    ),
    MOBILE_RECOVERY_EFFFORMER_L1_KD_CPU_4T(
        title = "EffFormer-L1+KD CPU-4T",
        useHflipTta = false,
        assetNames = listOf("tn5000_mobile_recovery_effformerl1_kd_mobile_recovery_s17.onnx"),
        inputSize = 224,
        runtimeProfile = OrtRuntimeProfile.CPU_4T,
    ),
    MOBILE_RECOVERY_EFFFORMER_L1_KD_XNNPACK_4T(
        title = "EffFormer-L1+KD XNNPACK-4T",
        useHflipTta = false,
        assetNames = listOf("tn5000_mobile_recovery_effformerl1_kd_mobile_recovery_s17.onnx"),
        inputSize = 224,
        runtimeProfile = OrtRuntimeProfile.XNNPACK_4T,
    ),
    MOBILE_RECOVERY_EFFFORMER_L1_KD_NNAPI(
        title = "EffFormer-L1+KD NNAPI",
        useHflipTta = false,
        assetNames = listOf("tn5000_mobile_recovery_effformerl1_kd_mobile_recovery_s17.onnx"),
        inputSize = 224,
        runtimeProfile = OrtRuntimeProfile.NNAPI,
    ),
    MOBILE_RECOVERY_EFFFORMER_L1_KD_NNAPI_FP16_NCHW(
        title = "EffFormer-L1+KD NNAPI-FP16-NCHW",
        useHflipTta = false,
        assetNames = listOf("tn5000_mobile_recovery_effformerl1_kd_mobile_recovery_s17.onnx"),
        inputSize = 224,
        runtimeProfile = OrtRuntimeProfile.NNAPI_FP16_NCHW,
    ),
    GGG_MCA_ORT_BASIC(
        title = "GGG-MCA ORT-BASIC",
        useHflipTta = false,
        assetNames = listOf("tn5000_ggg_mca_s27_current_ort_basic.onnx"),
    ),
    GGG_MCA_ORT_EXTENDED(
        title = "GGG-MCA ORT-EXT",
        useHflipTta = false,
        assetNames = listOf("tn5000_ggg_mca_s27_current_ort_extended.onnx"),
    ),
    GGG_MCA_DYNAMIC_INT8(
        title = "GGG-MCA dynamic-INT8",
        useHflipTta = false,
        assetNames = listOf("tn5000_ggg_mca_s27_current_dynamic_int8.onnx"),
    ),
    GGG_MCA_DYNAMIC_INT8_CPU_4T(
        title = "GGG-MCA dynamic-INT8 CPU-4T",
        useHflipTta = false,
        assetNames = listOf("tn5000_ggg_mca_s27_current_dynamic_int8.onnx"),
        runtimeProfile = OrtRuntimeProfile.CPU_4T,
    ),
    GGG_MCA_INPUT_128(
        title = "GGG-MCA input-128",
        useHflipTta = false,
        assetNames = listOf("tn5000_ggg_mca_s27_current_128.onnx"),
        inputSize = 128,
    ),
    GGG_MCA_CPU_1T(
        title = "GGG-MCA CPU-1T",
        useHflipTta = false,
        assetNames = listOf("tn5000_ggg_mca_s27_current.onnx"),
        runtimeProfile = OrtRuntimeProfile.CPU_1T,
    ),
    GGG_MCA_CPU_2T(
        title = "GGG-MCA CPU-2T",
        useHflipTta = false,
        assetNames = listOf("tn5000_ggg_mca_s27_current.onnx"),
        runtimeProfile = OrtRuntimeProfile.CPU_2T,
    ),
    GGG_MCA_CPU_4T(
        title = "GGG-MCA CPU-4T",
        useHflipTta = false,
        assetNames = listOf("tn5000_ggg_mca_s27_current.onnx"),
        runtimeProfile = OrtRuntimeProfile.CPU_4T,
    ),
    GGG_MCA_CPU_8T(
        title = "GGG-MCA CPU-8T",
        useHflipTta = false,
        assetNames = listOf("tn5000_ggg_mca_s27_current.onnx"),
        runtimeProfile = OrtRuntimeProfile.CPU_8T,
    ),
    GGG_MCA_XNNPACK_4T(
        title = "GGG-MCA XNNPACK-4T",
        useHflipTta = false,
        assetNames = listOf("tn5000_ggg_mca_s27_current.onnx"),
        runtimeProfile = OrtRuntimeProfile.XNNPACK_4T,
    ),
    GGG_MCA_NNAPI(
        title = "GGG-MCA NNAPI",
        useHflipTta = false,
        assetNames = listOf("tn5000_ggg_mca_s27_current.onnx"),
        runtimeProfile = OrtRuntimeProfile.NNAPI,
    ),
    GGG_MCA_NNAPI_FP16_NCHW(
        title = "GGG-MCA NNAPI-FP16-NCHW",
        useHflipTta = false,
        assetNames = listOf("tn5000_ggg_mca_s27_current.onnx"),
        runtimeProfile = OrtRuntimeProfile.NNAPI_FP16_NCHW,
    );
}

enum class InputPreparation {
    CENTER_CROP,
    DIRECT_RESIZE,
}

data class ModelAssetStat(
    val assetName: String,
    val sizeBytes: Long,
    val loadMs: Double,
    val newlyLoaded: Boolean,
)

data class PreparedMode(
    val mode: DeploymentMode,
    val assets: List<ModelAssetStat>,
) {
    val totalModelBytes: Long get() = assets.sumOf { it.sizeBytes }
    val modePrepareMs: Double get() = assets.sumOf { it.loadMs }
    val assetNames: List<String> get() = assets.map { it.assetName }
}

data class InferenceOptions(
    val mode: DeploymentMode,
    val threshold: Double,
    val temperature: Double,
    val inputPreparation: InputPreparation,
)

data class InferenceResult(
    val logits: FloatArray,
    val probabilities: FloatArray,
    val predictedIndex: Int,
    val predictedLabel: String,
    val preprocessMs: Double,
    val inferenceMs: Double,
    val totalMs: Double,
    val threshold: Double,
    val temperature: Double,
    val usedAssets: List<String>,
)

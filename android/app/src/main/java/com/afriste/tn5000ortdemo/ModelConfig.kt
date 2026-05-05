package com.afriste.tn5000ortdemo

enum class DeploymentMode(
    val title: String,
    val useHflipTta: Boolean,
    val assetNames: List<String>,
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

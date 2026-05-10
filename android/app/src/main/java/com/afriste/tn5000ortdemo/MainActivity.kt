package com.afriste.tn5000ortdemo

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.documentfile.provider.DocumentFile
import androidx.lifecycle.lifecycleScope
import com.afriste.tn5000ortdemo.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale

class MainActivity : ComponentActivity() {

    private lateinit var binding: ActivityMainBinding
    private var engine: Tn5000OrtEngine? = null
    private var currentBitmap: Bitmap? = null
    private var currentDatasetUri: Uri? = null

    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent(),
    ) { uri: Uri? ->
        if (uri != null) loadBitmapFromUri(uri)
    }

    private val pickDatasetFolderLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocumentTree(),
    ) { uri: Uri? ->
        if (uri == null) return@registerForActivityResult
        try {
            val flags = Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION
            contentResolver.takePersistableUriPermission(uri, flags)
        } catch (_: Exception) {
        }
        currentDatasetUri = uri
        val label = DocumentFile.fromTreeUri(this, uri)?.name ?: uri.toString()
        binding.tvDatasetStatus.text = "Dataset root: $label"
        binding.btnRunTestEval.isEnabled = engine != null
        binding.btnRunPaperBatch.isEnabled = engine != null
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnApplyTrainingPreset.setOnClickListener { applyTrainingAlignedPreset() }
        binding.btnPickDatasetFolder.setOnClickListener { pickDatasetFolderLauncher.launch(null) }
        binding.btnPickImage.setOnClickListener { pickImageLauncher.launch("image/*") }
        binding.btnRunInference.setOnClickListener { runSingleInference() }
        binding.btnRunTestEval.setOnClickListener { runTestEvaluation() }
        binding.btnRunPaperBatch.setOnClickListener { runPaperBatch() }
        binding.rgMode.setOnCheckedChangeListener { _, _ -> refreshModeHint() }

        applyTrainingAlignedPreset()
        restorePersistedDatasetUri()
        refreshModeHint()
        initEngine()
    }

    private fun restorePersistedDatasetUri() {
        val persisted = contentResolver.persistedUriPermissions
            .filter { it.isReadPermission }
            .mapNotNull { permission ->
                val doc = DocumentFile.fromTreeUri(this, permission.uri)
                val name = doc?.name.orEmpty()
                if (name == "TN5000_forReview") permission.uri to name else null
            }
            .firstOrNull()
            ?: return

        currentDatasetUri = persisted.first
        binding.tvDatasetStatus.text = "Dataset root: ${persisted.second}"
    }

    private fun applyTrainingAlignedPreset() {
        binding.etExpandRatio.setText("0.30")
        binding.etThreshold.setText("0.73")
        binding.etTemperature.setText("1.819830")
        binding.etColdRuns.setText("1")
        binding.etHotRuns.setText("1")
        binding.cbSquareCrop.isChecked = false
        binding.cbBatchFast.isChecked = false
        binding.cbBatchAccurate.isChecked = false
        binding.cbBatchEnsemble.isChecked = false
        binding.cbBatchGggMca.isChecked = false
        binding.cbBatchGggMcaTta.isChecked = false
        binding.cbBatchGggMcaEnsemble.isChecked = false
        binding.cbBatchMobileNetV3.isChecked = false
        binding.cbBatchRepVit.isChecked = false
        binding.cbBatchEfficientFormer.isChecked = false
        binding.cbBatchConvNeXt.isChecked = false
        binding.cbBatchSpeedCpuSweep.isChecked = true
        binding.cbBatchSpeedQuant.isChecked = false
        binding.cbBatchSpeedInput128.isChecked = false
        binding.cbBatchSpeedXnnpack.isChecked = false
        binding.cbBatchSpeedNnapi.isChecked = false
        binding.cbBatchSpeedNnapiFp16.isChecked = false
    }

    private fun currentMode(): DeploymentMode {
        return when (binding.rgMode.checkedRadioButtonId) {
            R.id.rbModeFast -> DeploymentMode.FAST
            R.id.rbModeAccurate -> DeploymentMode.ACCURATE
            R.id.rbModeEnsemble -> DeploymentMode.ENSEMBLE
            R.id.rbModeGggMca -> DeploymentMode.GGG_MCA
            R.id.rbModeGggMcaTta -> DeploymentMode.GGG_MCA_TTA
            R.id.rbModeGggMcaEnsemble -> DeploymentMode.GGG_MCA_ENSEMBLE
            else -> DeploymentMode.FAST
        }
    }

    private fun selectedBatchModes(): List<DeploymentMode> {
        val modes = mutableListOf<DeploymentMode>()
        if (binding.cbBatchFast.isChecked) modes += DeploymentMode.FAST
        if (binding.cbBatchAccurate.isChecked) modes += DeploymentMode.ACCURATE
        if (binding.cbBatchEnsemble.isChecked) modes += DeploymentMode.ENSEMBLE
        if (binding.cbBatchGggMca.isChecked) modes += DeploymentMode.GGG_MCA
        if (binding.cbBatchGggMcaTta.isChecked) modes += DeploymentMode.GGG_MCA_TTA
        if (binding.cbBatchGggMcaEnsemble.isChecked) modes += DeploymentMode.GGG_MCA_ENSEMBLE
        if (binding.cbBatchMobileNetV3.isChecked) modes += DeploymentMode.BASELINE_MOBILENETV3
        if (binding.cbBatchRepVit.isChecked) modes += DeploymentMode.BASELINE_REPVIT_M11
        if (binding.cbBatchEfficientFormer.isChecked) modes += DeploymentMode.BASELINE_EFFICIENTFORMER_L1
        if (binding.cbBatchConvNeXt.isChecked) modes += DeploymentMode.BASELINE_CONVNEXT_TINY
        if (binding.cbBatchSpeedCpuSweep.isChecked) {
            modes += listOf(
                DeploymentMode.GGG_MCA,
                DeploymentMode.GGG_MCA_ORT_BASIC,
                DeploymentMode.GGG_MCA_ORT_EXTENDED,
                DeploymentMode.GGG_MCA_CPU_1T,
                DeploymentMode.GGG_MCA_CPU_2T,
                DeploymentMode.GGG_MCA_CPU_4T,
                DeploymentMode.GGG_MCA_CPU_8T,
            )
        }
        if (binding.cbBatchSpeedQuant.isChecked) {
            modes += listOf(
                DeploymentMode.GGG_MCA_DYNAMIC_INT8,
                DeploymentMode.GGG_MCA_DYNAMIC_INT8_CPU_4T,
            )
        }
        if (binding.cbBatchSpeedInput128.isChecked) modes += DeploymentMode.GGG_MCA_INPUT_128
        if (binding.cbBatchSpeedXnnpack.isChecked) modes += DeploymentMode.GGG_MCA_XNNPACK_4T
        if (binding.cbBatchSpeedNnapi.isChecked) modes += DeploymentMode.GGG_MCA_NNAPI
        if (binding.cbBatchSpeedNnapiFp16.isChecked) modes += DeploymentMode.GGG_MCA_NNAPI_FP16_NCHW
        return modes
    }

    private fun refreshModeHint() {
        val mode = currentMode()
        binding.tvModeHint.text = when (mode) {
            DeploymentMode.FAST -> "Fast = single ONNX + threshold + temperature scaling. Lowest latency."
            DeploymentMode.ACCURATE -> "Accurate = single ONNX + HFlip TTA + threshold + temperature scaling."
            DeploymentMode.ENSEMBLE -> "Ensemble = 3 ONNX checkpoints + HFlip TTA + threshold + temperature scaling. Best fidelity, highest cost."
            DeploymentMode.GGG_MCA -> "GGG-MCA = current withMCA seed27 ONNX. Use threshold 0.73 and temperature 1.819830."
            DeploymentMode.GGG_MCA_TTA -> "GGG-MCA+TTA = current withMCA seed27 ONNX with horizontal flip test-time augmentation."
            DeploymentMode.GGG_MCA_ENSEMBLE -> "GGG-MCA Ensemble = current top-3 checkpoints with horizontal flip TTA."
            DeploymentMode.BASELINE_MOBILENETV3 -> "MobileNetV3 baseline = seed27 ONNX, single-pass latency mode."
            DeploymentMode.BASELINE_REPVIT_M11 -> "RepViT-M1.1 baseline = seed27 ONNX, single-pass latency mode."
            DeploymentMode.BASELINE_EFFICIENTFORMER_L1 -> "EfficientFormer-L1 baseline = seed27 ONNX at 224 input."
            DeploymentMode.BASELINE_CONVNEXT_TINY -> "ConvNeXt-Tiny baseline = seed27 ONNX, single-pass latency mode."
            else -> "${mode.title} = speed-profile benchmark mode."
        }
    }

    private fun initEngine() {
        lifecycleScope.launch {
            binding.tvModelStatus.text = "Model assets: checking..."
            try {
                val loaded = withContext(Dispatchers.Default) { Tn5000OrtEngine(this@MainActivity) }
                engine = loaded
                binding.tvModelStatus.text = "Model assets status:\n${loaded.summarizeAvailability()}"
                binding.btnRunTestEval.isEnabled = currentDatasetUri != null
                binding.btnRunPaperBatch.isEnabled = currentDatasetUri != null
            } catch (e: Exception) {
                binding.tvModelStatus.text = "Model init failed"
                binding.tvResult.text = e.stackTraceToString()
            }
        }
    }

    private fun currentOptions(forSingleImage: Boolean): InferenceOptions {
        val threshold = binding.etThreshold.text?.toString()?.trim()?.toDoubleOrNull() ?: 0.61
        val temperature = binding.etTemperature.text?.toString()?.trim()?.toDoubleOrNull()?.coerceAtLeast(1e-6) ?: 1.157835
        return InferenceOptions(
            mode = currentMode(),
            threshold = threshold,
            temperature = temperature,
            inputPreparation = if (forSingleImage) InputPreparation.CENTER_CROP else InputPreparation.DIRECT_RESIZE,
        )
    }

    private fun currentEvalOptions(): EvaluationOptions {
        val threshold = binding.etThreshold.text?.toString()?.trim()?.toDoubleOrNull() ?: 0.61
        val temperature = binding.etTemperature.text?.toString()?.trim()?.toDoubleOrNull()?.coerceAtLeast(1e-6) ?: 1.157835
        val expandRatio = binding.etExpandRatio.text?.toString()?.trim()?.toFloatOrNull() ?: 0.30f
        return EvaluationOptions(
            mode = currentMode(),
            threshold = threshold,
            temperature = temperature,
            expandRatio = expandRatio,
            squareCrop = binding.cbSquareCrop.isChecked,
        )
    }

    private fun currentBatchOptions(): PaperBatchOptions {
        val coldRuns = binding.etColdRuns.text?.toString()?.trim()?.toIntOrNull() ?: 1
        val hotRuns = binding.etHotRuns.text?.toString()?.trim()?.toIntOrNull() ?: 3
        return PaperBatchOptions(
            selectedModes = selectedBatchModes(),
            coldRunsPerMode = coldRuns,
            hotRunsPerMode = hotRuns,
            note = binding.etBatchNote.text?.toString()?.trim().orEmpty(),
        )
    }

    private fun loadBitmapFromUri(uri: Uri) {
        lifecycleScope.launch {
            try {
                val bitmap = withContext(Dispatchers.IO) {
                    contentResolver.openInputStream(uri).use { input ->
                        requireNotNull(input) { "Cannot open image stream." }
                        BitmapFactory.decodeStream(input) ?: error("Failed to decode bitmap.")
                    }
                }
                currentBitmap = bitmap
                binding.ivPreview.setImageBitmap(bitmap)
                binding.tvImageStatus.text = "Image: ${bitmap.width} x ${bitmap.height}"
                binding.btnRunInference.isEnabled = true
                binding.tvResult.text = "Image ready. Single-image mode uses center crop + resize."
                binding.tvLogits.text = "logits: -"
                binding.tvProbabilities.text = "probabilities: -"
                binding.tvLatency.text = "latency: -"
            } catch (e: Exception) {
                Toast.makeText(this@MainActivity, e.message ?: "Failed to load image.", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun setBusyUi(running: Boolean) {
        binding.btnRunInference.isEnabled = !running && currentBitmap != null
        binding.btnRunTestEval.isEnabled = !running && engine != null && currentDatasetUri != null
        binding.btnRunPaperBatch.isEnabled = !running && engine != null && currentDatasetUri != null
        binding.btnPickDatasetFolder.isEnabled = !running
        binding.btnPickImage.isEnabled = !running
        binding.btnApplyTrainingPreset.isEnabled = !running
    }

    private fun runSingleInference() {
        val bitmap = currentBitmap
        val ort = engine
        if (bitmap == null || ort == null) {
            Toast.makeText(this, "Model or image is not ready.", Toast.LENGTH_SHORT).show()
            return
        }

        lifecycleScope.launch {
            binding.btnRunInference.isEnabled = false
            binding.tvResult.text = "Running single-image inference..."
            try {
                val options = currentOptions(forSingleImage = true)
                val result = withContext(Dispatchers.Default) {
                    ort.infer(bitmap, options)
                }
                binding.tvResult.text = buildString {
                    appendLine("Prediction: ${result.predictedLabel}")
                    appendLine("mode = ${options.mode.title}")
                    appendLine("threshold = ${"%.4f".format(result.threshold)}")
                    appendLine("temperature = ${"%.6f".format(result.temperature)}")
                    appendLine("assets = ${result.usedAssets.joinToString()}")
                }
                binding.tvLogits.text = "logits: [${result.logits.joinToString { "%.5f".format(it) }}]"
                binding.tvProbabilities.text = "probabilities: [${result.probabilities.joinToString { "%.5f".format(it) }}]"
                binding.tvLatency.text = "preprocess=%.2f ms | inference=%.2f ms | total=%.2f ms".format(
                    result.preprocessMs,
                    result.inferenceMs,
                    result.totalMs,
                )
            } catch (e: Exception) {
                binding.tvResult.text = e.stackTraceToString()
            } finally {
                binding.btnRunInference.isEnabled = true
            }
        }
    }

    private fun runTestEvaluation() {
        val ort = engine
        val datasetUri = currentDatasetUri
        if (ort == null) {
            Toast.makeText(this, "Model engine is not ready.", Toast.LENGTH_SHORT).show()
            return
        }
        if (datasetUri == null) {
            Toast.makeText(this, "Please select TN5000_forReview folder first.", Toast.LENGTH_SHORT).show()
            return
        }

        lifecycleScope.launch {
            val options = currentEvalOptions()
            setBusyUi(true)
            binding.progressEval.progress = 0
            binding.tvEvalProgress.text = "Evaluation: starting..."
            binding.tvEvalSummary.text = "Summary: running..."
            binding.tvBatchSummary.text = "Batch summary: -"

            try {
                val summary = withContext(Dispatchers.Default) {
                    Tn5000DatasetEvaluator(this@MainActivity, ort).evaluateTestSet(
                        rootUri = datasetUri,
                        options = options,
                    ) { processed, total, currentId, roiBitmap ->
                        runOnUiThread {
                            binding.progressEval.max = total
                            binding.progressEval.progress = processed
                            binding.tvEvalProgress.text = "Evaluation: $processed / $total | current = $currentId"
                            if (roiBitmap != null) {
                                binding.ivPreview.setImageBitmap(roiBitmap)
                                binding.tvImageStatus.text = "ROI preview from test set: $currentId"
                            }
                        }
                    }
                }
                binding.tvEvalSummary.text = buildSummaryText(summary)
                binding.tvResult.text = buildString {
                    appendLine("Test-set evaluation finished.")
                    appendLine("CSV: ${summary.csvFile.absolutePath}")
                    appendLine("TXT: ${summary.summaryFile.absolutePath}")
                }
            } catch (e: Exception) {
                binding.tvEvalSummary.text = "Summary: failed"
                binding.tvResult.text = e.stackTraceToString()
            } finally {
                setBusyUi(false)
            }
        }
    }

    private fun runPaperBatch() {
        val ort = engine
        val datasetUri = currentDatasetUri
        if (ort == null) {
            Toast.makeText(this, "Model engine is not ready.", Toast.LENGTH_SHORT).show()
            return
        }
        if (datasetUri == null) {
            Toast.makeText(this, "Please select TN5000_forReview folder first.", Toast.LENGTH_SHORT).show()
            return
        }
        val batchOptions = currentBatchOptions()
        if (batchOptions.selectedModes.isEmpty()) {
            Toast.makeText(this, "Select at least one mode for the paper batch.", Toast.LENGTH_SHORT).show()
            return
        }

        lifecycleScope.launch {
            val baseOptions = currentEvalOptions()
            setBusyUi(true)
            binding.progressEval.progress = 0
            binding.tvEvalProgress.text = "Batch: starting..."
            binding.tvEvalSummary.text = "Summary: -"
            binding.tvBatchSummary.text = "Batch summary: running..."

            try {
                val batchSummary = withContext(Dispatchers.Default) {
                    PaperBatchRunner(this@MainActivity, ort).runPaperBatch(
                        rootUri = datasetUri,
                        baseOptions = baseOptions,
                        batchOptions = batchOptions,
                    ) { batchIndex, batchTotal, message, evalProcessed, evalTotal, currentId ->
                        runOnUiThread {
                            val currentRun = (batchIndex + 1).coerceAtMost(batchTotal)
                            binding.progressEval.max = if (evalTotal > 0) evalTotal else 1
                            binding.progressEval.progress = if (evalTotal > 0) evalProcessed else 0
                            binding.tvEvalProgress.text = buildString {
                                append("Batch run $currentRun / $batchTotal")
                                append(" | $message")
                                if (currentId != null && evalTotal > 0) {
                                    append(" | eval $evalProcessed / $evalTotal | current = $currentId")
                                }
                            }
                        }
                    }
                }
                binding.tvBatchSummary.text = buildBatchSummaryText(batchSummary)
                binding.tvResult.text = buildString {
                    appendLine("Paper batch finished.")
                    appendLine("batch_dir = ${batchSummary.batchDir.absolutePath}")
                    appendLine("manifest_csv = ${batchSummary.runManifestCsv.absolutePath}")
                    appendLine("aggregate_by_mode = ${batchSummary.aggregateByModeCsv.absolutePath}")
                    appendLine("aggregate_by_mode_phase = ${batchSummary.aggregateByModePhaseCsv.absolutePath}")
                    appendLine("summary_txt = ${batchSummary.summaryTxt.absolutePath}")
                    appendLine("device_info = ${batchSummary.deviceInfoTxt.absolutePath}")
                }
            } catch (e: Exception) {
                binding.tvBatchSummary.text = "Batch summary: failed"
                binding.tvResult.text = e.stackTraceToString()
            } finally {
                setBusyUi(false)
            }
        }
    }

    private fun buildSummaryText(summary: EvaluationSummary): String {
        return buildString {
            appendLine("Summary:")
            appendLine("mode = ${summary.mode.title}")
            appendLine("assets = ${summary.assetsUsed.joinToString()}")
            appendLine("model_MB = ${"%.2f".format(summary.modelBytesTotal / (1024.0 * 1024.0))}")
            appendLine("mode_prepare_ms = ${"%.2f".format(summary.modePrepareMs)}")
            appendLine()
            appendLine("total = ${summary.total}")
            appendLine("correct = ${summary.correct}")
            appendLine("accuracy = ${"%.4f".format(summary.accuracy)}")
            appendLine("balanced_accuracy = ${"%.4f".format(summary.balancedAccuracy)}")
            appendLine("precision_macro = ${"%.4f".format(summary.precisionMacro)}")
            appendLine("recall_macro = ${"%.4f".format(summary.recallMacro)}")
            appendLine("f1_macro = ${"%.4f".format(summary.f1Macro)}")
            appendLine("auc = ${"%.4f".format(summary.auc)}")
            appendLine("recall_benign(0) = ${"%.4f".format(summary.recallBenign)}")
            appendLine("recall_malignant(1) = ${"%.4f".format(summary.recallMalignant)}")
            appendLine("confusion = [[TN=${summary.tn}, FP=${summary.fp}], [FN=${summary.fn}, TP=${summary.tp}]]")
            appendLine()
            appendLine("avg_preprocess_ms = ${"%.2f".format(summary.avgPreprocessMs)}")
            appendLine("avg_inference_ms = ${"%.2f".format(summary.avgInferenceMs)}")
            appendLine("avg_total_ms = ${"%.2f".format(summary.avgTotalMs)}")
            appendLine("p50_total_ms = ${"%.2f".format(summary.p50TotalMs)}")
            appendLine("p95_total_ms = ${"%.2f".format(summary.p95TotalMs)}")
            appendLine("total_runtime_sec = ${"%.2f".format(summary.totalRuntimeSec)}")
            appendLine("peak_java_heap_mb = ${"%.2f".format(summary.peakJavaHeapMb)}")
            appendLine("peak_native_heap_mb = ${"%.2f".format(summary.peakNativeHeapMb)}")
            appendLine()
            appendLine("csv = ${summary.csvFile.absolutePath}")
            appendLine("txt = ${summary.summaryFile.absolutePath}")
        }
    }

    private fun buildBatchSummaryText(batchSummary: PaperBatchSummary): String {
        val grouped = batchSummary.records.groupBy { it.mode }
        return buildString {
            appendLine("Batch summary:")
            appendLine("batch_dir = ${batchSummary.batchDir.absolutePath}")
            appendLine("finished_runs = ${batchSummary.finishedRuns} / ${batchSummary.totalRuns}")
            appendLine()
            DeploymentMode.entries.forEach { mode ->
                val records = grouped[mode].orEmpty()
                if (records.isEmpty()) return@forEach
                appendLine("[${mode.title}]")
                appendLine("runs = ${records.size}")
                appendLine("accuracy_mean = ${formatStat(records.map { it.summary.accuracy })}")
                appendLine("bal_acc_mean = ${formatStat(records.map { it.summary.balancedAccuracy })}")
                appendLine("f1_mean = ${formatStat(records.map { it.summary.f1Macro })}")
                appendLine("auc_mean = ${formatStat(records.map { it.summary.auc })}")
                appendLine("avg_total_ms_mean = ${formatStat(records.map { it.summary.avgTotalMs })}")
                appendLine()
            }
            appendLine("manifest_csv = ${batchSummary.runManifestCsv.absolutePath}")
            appendLine("aggregate_by_mode = ${batchSummary.aggregateByModeCsv.absolutePath}")
            appendLine("aggregate_by_mode_phase = ${batchSummary.aggregateByModePhaseCsv.absolutePath}")
            appendLine("summary_txt = ${batchSummary.summaryTxt.absolutePath}")
            appendLine("device_info = ${batchSummary.deviceInfoTxt.absolutePath}")
        }
    }

    private fun formatStat(values: List<Double>): String {
        if (values.isEmpty()) return "-"
        val mean = values.average()
        val variance = values.map { (it - mean) * (it - mean) }.average()
        val std = kotlin.math.sqrt(variance)
        return String.format(Locale.US, "%.4f ± %.4f", mean, std)
    }

    override fun onDestroy() {
        engine?.close()
        engine = null
        super.onDestroy()
    }
}

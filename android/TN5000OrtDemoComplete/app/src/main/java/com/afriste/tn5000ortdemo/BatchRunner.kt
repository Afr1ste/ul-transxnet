package com.afriste.tn5000ortdemo

import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Debug
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.sqrt

data class PaperBatchOptions(
    val selectedModes: List<DeploymentMode>,
    val coldRunsPerMode: Int,
    val hotRunsPerMode: Int,
    val note: String,
)

data class PaperBatchRunRecord(
    val mode: DeploymentMode,
    val phase: String,
    val phaseIndex: Int,
    val summary: EvaluationSummary,
)

data class PaperBatchSummary(
    val batchDir: File,
    val runManifestCsv: File,
    val aggregateByModeCsv: File,
    val aggregateByModePhaseCsv: File,
    val summaryTxt: File,
    val deviceInfoTxt: File,
    val totalRuns: Int,
    val finishedRuns: Int,
    val records: List<PaperBatchRunRecord>,
)

class PaperBatchRunner(
    private val context: Context,
    private val engine: Tn5000OrtEngine,
) {
    private val evaluator = Tn5000DatasetEvaluator(context, engine)

    fun runPaperBatch(
        rootUri: Uri,
        baseOptions: EvaluationOptions,
        batchOptions: PaperBatchOptions,
        progressCallback: ((batchIndex: Int, batchTotal: Int, message: String, evalProcessed: Int, evalTotal: Int, currentId: String?) -> Unit)? = null,
    ): PaperBatchSummary {
        require(batchOptions.selectedModes.isNotEmpty()) { "Select at least one deployment mode." }
        require(batchOptions.coldRunsPerMode >= 0) { "coldRunsPerMode must be >= 0" }
        require(batchOptions.hotRunsPerMode >= 0) { "hotRunsPerMode must be >= 0" }
        require(batchOptions.coldRunsPerMode + batchOptions.hotRunsPerMode > 0) { "Total runs per mode must be > 0" }

        val batchDir = createBatchDir()
        val records = mutableListOf<PaperBatchRunRecord>()
        val totalRuns = batchOptions.selectedModes.sumOf { batchOptions.coldRunsPerMode + batchOptions.hotRunsPerMode }
        var completedRuns = 0

        writeDeviceInfo(batchDir, batchOptions, baseOptions)

        batchOptions.selectedModes.forEach { mode ->
            repeat(batchOptions.coldRunsPerMode) { idx ->
                engine.clearSessionCache()
                System.gc()
                val phase = "cold"
                val runNumber = idx + 1
                val stem = "${mode.title.lowercase(Locale.US)}_${phase}${runNumber}"
                progressCallback?.invoke(completedRuns, totalRuns, "Preparing ${mode.title} $phase run $runNumber", 0, 0, null)
                val summary = evaluator.evaluateTestSet(
                    rootUri = rootUri,
                    options = baseOptions.copy(mode = mode),
                    outputDir = batchDir,
                    fileStem = stem,
                ) { processed, total, currentId, _ ->
                    progressCallback?.invoke(completedRuns, totalRuns, "Running ${mode.title} $phase run $runNumber", processed, total, currentId)
                }
                records += PaperBatchRunRecord(mode = mode, phase = phase, phaseIndex = runNumber, summary = summary)
                completedRuns += 1
            }

            repeat(batchOptions.hotRunsPerMode) { idx ->
                val phase = "hot"
                val runNumber = idx + 1
                val stem = "${mode.title.lowercase(Locale.US)}_${phase}${runNumber}"
                progressCallback?.invoke(completedRuns, totalRuns, "Preparing ${mode.title} $phase run $runNumber", 0, 0, null)
                val summary = evaluator.evaluateTestSet(
                    rootUri = rootUri,
                    options = baseOptions.copy(mode = mode),
                    outputDir = batchDir,
                    fileStem = stem,
                ) { processed, total, currentId, _ ->
                    progressCallback?.invoke(completedRuns, totalRuns, "Running ${mode.title} $phase run $runNumber", processed, total, currentId)
                }
                records += PaperBatchRunRecord(mode = mode, phase = phase, phaseIndex = runNumber, summary = summary)
                completedRuns += 1
            }
        }

        val manifestCsv = File(batchDir, "batch_runs_manifest.csv")
        val aggregateModeCsv = File(batchDir, "aggregate_by_mode.csv")
        val aggregateModePhaseCsv = File(batchDir, "aggregate_by_mode_phase.csv")
        val summaryTxt = File(batchDir, "batch_summary.txt")
        val deviceInfoTxt = File(batchDir, "device_info.txt")

        writeRunManifest(records, manifestCsv)
        writeAggregateByMode(records, aggregateModeCsv)
        writeAggregateByModePhase(records, aggregateModePhaseCsv)
        writeBatchSummary(records, batchOptions, baseOptions, summaryTxt)

        progressCallback?.invoke(completedRuns, totalRuns, "Paper batch finished", 0, 0, null)
        return PaperBatchSummary(
            batchDir = batchDir,
            runManifestCsv = manifestCsv,
            aggregateByModeCsv = aggregateModeCsv,
            aggregateByModePhaseCsv = aggregateModePhaseCsv,
            summaryTxt = summaryTxt,
            deviceInfoTxt = deviceInfoTxt,
            totalRuns = totalRuns,
            finishedRuns = completedRuns,
            records = records,
        )
    }

    private fun createBatchDir(): File {
        val root = File(context.getExternalFilesDir(null), "eval_reports")
        root.mkdirs()
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        return File(root, "paper_batch_$stamp").apply { mkdirs() }
    }

    private fun writeDeviceInfo(batchDir: File, batchOptions: PaperBatchOptions, baseOptions: EvaluationOptions) {
        val file = File(batchDir, "device_info.txt")
        FileWriter(file).use { writer ->
            writer.appendLine("TN5000 Android paper batch device info")
            writer.appendLine("========================================")
            writer.appendLine("manufacturer=${Build.MANUFACTURER}")
            writer.appendLine("brand=${Build.BRAND}")
            writer.appendLine("model=${Build.MODEL}")
            writer.appendLine("device=${Build.DEVICE}")
            writer.appendLine("product=${Build.PRODUCT}")
            writer.appendLine("sdk_int=${Build.VERSION.SDK_INT}")
            writer.appendLine("release=${Build.VERSION.RELEASE}")
            writer.appendLine("abis=${Build.SUPPORTED_ABIS.joinToString(";")}")
            writer.appendLine("native_heap_start_mb=${"%.4f".format(Locale.US, Debug.getNativeHeapAllocatedSize() / (1024.0 * 1024.0))}")
            writer.appendLine("")
            writer.appendLine("selected_modes=${batchOptions.selectedModes.joinToString(",") { it.title }}")
            writer.appendLine("cold_runs_per_mode=${batchOptions.coldRunsPerMode}")
            writer.appendLine("hot_runs_per_mode=${batchOptions.hotRunsPerMode}")
            writer.appendLine("threshold=${"%.6f".format(Locale.US, baseOptions.threshold)}")
            writer.appendLine("temperature=${"%.6f".format(Locale.US, baseOptions.temperature)}")
            writer.appendLine("expand_ratio=${"%.6f".format(Locale.US, baseOptions.expandRatio.toDouble())}")
            writer.appendLine("square_crop=${baseOptions.squareCrop}")
            writer.appendLine("note=${batchOptions.note}")
        }
    }

    private fun writeRunManifest(records: List<PaperBatchRunRecord>, outFile: File) {
        FileWriter(outFile).use { writer ->
            writer.appendLine(
                listOf(
                    "mode", "phase", "phase_index", "total", "correct", "accuracy", "balanced_accuracy",
                    "precision_macro", "recall_macro", "f1_macro", "auc", "recall_benign", "recall_malignant",
                    "tn", "fp", "fn", "tp", "avg_preprocess_ms", "avg_inference_ms", "avg_total_ms",
                    "p50_total_ms", "p95_total_ms", "total_runtime_sec", "peak_java_heap_mb", "peak_native_heap_mb",
                    "model_mb", "mode_prepare_ms", "csv_path", "summary_path"
                ).joinToString(",")
            )
            records.forEach { record ->
                val s = record.summary
                writer.appendLine(
                    listOf(
                        record.mode.title,
                        record.phase,
                        record.phaseIndex.toString(),
                        s.total.toString(),
                        s.correct.toString(),
                        fmt(s.accuracy),
                        fmt(s.balancedAccuracy),
                        fmt(s.precisionMacro),
                        fmt(s.recallMacro),
                        fmt(s.f1Macro),
                        fmt(s.auc),
                        fmt(s.recallBenign),
                        fmt(s.recallMalignant),
                        s.tn.toString(),
                        s.fp.toString(),
                        s.fn.toString(),
                        s.tp.toString(),
                        fmt(s.avgPreprocessMs),
                        fmt(s.avgInferenceMs),
                        fmt(s.avgTotalMs),
                        fmt(s.p50TotalMs),
                        fmt(s.p95TotalMs),
                        fmt(s.totalRuntimeSec),
                        fmt(s.peakJavaHeapMb),
                        fmt(s.peakNativeHeapMb),
                        fmt(s.modelBytesTotal / (1024.0 * 1024.0)),
                        fmt(s.modePrepareMs),
                        csvEscape(s.csvFile.absolutePath),
                        csvEscape(s.summaryFile.absolutePath),
                    ).joinToString(",")
                )
            }
        }
    }

    private fun writeAggregateByMode(records: List<PaperBatchRunRecord>, outFile: File) {
        val grouped = records.groupBy { it.mode }
        FileWriter(outFile).use { writer ->
            writer.appendLine("mode,n_runs,accuracy_mean,accuracy_std,balanced_accuracy_mean,balanced_accuracy_std,f1_macro_mean,f1_macro_std,auc_mean,auc_std,avg_total_ms_mean,avg_total_ms_std,p95_total_ms_mean,p95_total_ms_std,peak_java_heap_mb_mean,peak_native_heap_mb_mean")
            grouped.entries.sortedBy { it.key.ordinal }.forEach { (mode, modeRecords) ->
                writer.appendLine(
                    listOf(
                        mode.title,
                        modeRecords.size.toString(),
                        fmt(meanOf(modeRecords) { it.summary.accuracy }),
                        fmt(stdOf(modeRecords) { it.summary.accuracy }),
                        fmt(meanOf(modeRecords) { it.summary.balancedAccuracy }),
                        fmt(stdOf(modeRecords) { it.summary.balancedAccuracy }),
                        fmt(meanOf(modeRecords) { it.summary.f1Macro }),
                        fmt(stdOf(modeRecords) { it.summary.f1Macro }),
                        fmt(meanOf(modeRecords) { it.summary.auc }),
                        fmt(stdOf(modeRecords) { it.summary.auc }),
                        fmt(meanOf(modeRecords) { it.summary.avgTotalMs }),
                        fmt(stdOf(modeRecords) { it.summary.avgTotalMs }),
                        fmt(meanOf(modeRecords) { it.summary.p95TotalMs }),
                        fmt(stdOf(modeRecords) { it.summary.p95TotalMs }),
                        fmt(meanOf(modeRecords) { it.summary.peakJavaHeapMb }),
                        fmt(meanOf(modeRecords) { it.summary.peakNativeHeapMb }),
                    ).joinToString(",")
                )
            }
        }
    }

    private fun writeAggregateByModePhase(records: List<PaperBatchRunRecord>, outFile: File) {
        val grouped = records.groupBy { Pair(it.mode, it.phase) }
        FileWriter(outFile).use { writer ->
            writer.appendLine("mode,phase,n_runs,accuracy_mean,balanced_accuracy_mean,f1_macro_mean,auc_mean,avg_total_ms_mean,p95_total_ms_mean,mode_prepare_ms_mean")
            grouped.entries.sortedWith(compareBy({ it.key.first.ordinal }, { it.key.second })).forEach { (key, group) ->
                writer.appendLine(
                    listOf(
                        key.first.title,
                        key.second,
                        group.size.toString(),
                        fmt(meanOf(group) { it.summary.accuracy }),
                        fmt(meanOf(group) { it.summary.balancedAccuracy }),
                        fmt(meanOf(group) { it.summary.f1Macro }),
                        fmt(meanOf(group) { it.summary.auc }),
                        fmt(meanOf(group) { it.summary.avgTotalMs }),
                        fmt(meanOf(group) { it.summary.p95TotalMs }),
                        fmt(meanOf(group) { it.summary.modePrepareMs }),
                    ).joinToString(",")
                )
            }
        }
    }

    private fun writeBatchSummary(
        records: List<PaperBatchRunRecord>,
        batchOptions: PaperBatchOptions,
        baseOptions: EvaluationOptions,
        outFile: File,
    ) {
        FileWriter(outFile).use { writer ->
            writer.appendLine("TN5000 Android paper batch summary")
            writer.appendLine("========================================")
            writer.appendLine("selected_modes=${batchOptions.selectedModes.joinToString(",") { it.title }}")
            writer.appendLine("cold_runs_per_mode=${batchOptions.coldRunsPerMode}")
            writer.appendLine("hot_runs_per_mode=${batchOptions.hotRunsPerMode}")
            writer.appendLine("threshold=${fmt(baseOptions.threshold)}")
            writer.appendLine("temperature=${fmt(baseOptions.temperature)}")
            writer.appendLine("expand_ratio=${fmt(baseOptions.expandRatio.toDouble())}")
            writer.appendLine("square_crop=${baseOptions.squareCrop}")
            writer.appendLine("note=${batchOptions.note}")
            writer.appendLine("")
            batchOptions.selectedModes.forEach { mode ->
                val modeRecords = records.filter { it.mode == mode }
                if (modeRecords.isEmpty()) return@forEach
                writer.appendLine("[${mode.title}]")
                writer.appendLine("runs=${modeRecords.size}")
                writer.appendLine("accuracy_mean=${fmt(meanOf(modeRecords) { it.summary.accuracy })}")
                writer.appendLine("accuracy_std=${fmt(stdOf(modeRecords) { it.summary.accuracy })}")
                writer.appendLine("balanced_accuracy_mean=${fmt(meanOf(modeRecords) { it.summary.balancedAccuracy })}")
                writer.appendLine("balanced_accuracy_std=${fmt(stdOf(modeRecords) { it.summary.balancedAccuracy })}")
                writer.appendLine("f1_macro_mean=${fmt(meanOf(modeRecords) { it.summary.f1Macro })}")
                writer.appendLine("f1_macro_std=${fmt(stdOf(modeRecords) { it.summary.f1Macro })}")
                writer.appendLine("auc_mean=${fmt(meanOf(modeRecords) { it.summary.auc })}")
                writer.appendLine("auc_std=${fmt(stdOf(modeRecords) { it.summary.auc })}")
                writer.appendLine("avg_total_ms_mean=${fmt(meanOf(modeRecords) { it.summary.avgTotalMs })}")
                writer.appendLine("p95_total_ms_mean=${fmt(meanOf(modeRecords) { it.summary.p95TotalMs })}")
                writer.appendLine("cold_runs=${modeRecords.count { it.phase == "cold" }}")
                writer.appendLine("hot_runs=${modeRecords.count { it.phase == "hot" }}")
                writer.appendLine("")
            }
        }
    }

    private fun meanOf(records: List<PaperBatchRunRecord>, selector: (PaperBatchRunRecord) -> Double): Double {
        if (records.isEmpty()) return 0.0
        return records.map(selector).average()
    }

    private fun stdOf(records: List<PaperBatchRunRecord>, selector: (PaperBatchRunRecord) -> Double): Double {
        if (records.isEmpty()) return 0.0
        val values = records.map(selector)
        val mean = values.average()
        val variance = values.map { (it - mean) * (it - mean) }.average()
        return sqrt(variance)
    }

    private fun fmt(value: Double): String = "%.6f".format(Locale.US, value)

    private fun csvEscape(value: String): String {
        val needsQuote = value.contains(',') || value.contains('"') || value.contains('\n')
        if (!needsQuote) return value
        return "\"${value.replace("\"", "\"\"")}\""
    }
}

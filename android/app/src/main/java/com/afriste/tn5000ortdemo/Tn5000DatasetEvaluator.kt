package com.afriste.tn5000ortdemo

import android.content.ContentResolver
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Debug
import androidx.documentfile.provider.DocumentFile
import java.io.File
import java.io.FileWriter
import kotlin.math.max

private fun safeDiv(numerator: Long, denominator: Long): Double {
    return if (denominator == 0L) 0.0 else numerator.toDouble() / denominator.toDouble()
}

data class EvaluationOptions(
    val mode: DeploymentMode,
    val threshold: Double,
    val temperature: Double,
    val expandRatio: Float,
    val squareCrop: Boolean,
)

data class SamplePrediction(
    val sampleId: String,
    val groundTruth: Int,
    val predicted: Int,
    val probability0: Double,
    val probability1: Double,
    val preprocessMs: Double,
    val inferenceMs: Double,
    val totalMs: Double,
    val bbox: VocBBox,
)

data class EvaluationSummary(
    val mode: DeploymentMode,
    val assetsUsed: List<String>,
    val modelBytesTotal: Long,
    val modePrepareMs: Double,
    val total: Int,
    val correct: Int,
    val accuracy: Double,
    val balancedAccuracy: Double,
    val precisionMacro: Double,
    val recallMacro: Double,
    val f1Macro: Double,
    val auc: Double,
    val recallBenign: Double,
    val recallMalignant: Double,
    val tn: Int,
    val fp: Int,
    val fn: Int,
    val tp: Int,
    val avgPreprocessMs: Double,
    val avgInferenceMs: Double,
    val avgTotalMs: Double,
    val p50TotalMs: Double,
    val p95TotalMs: Double,
    val totalRuntimeSec: Double,
    val peakJavaHeapMb: Double,
    val peakNativeHeapMb: Double,
    val csvFile: File,
    val summaryFile: File,
)

class Tn5000DatasetEvaluator(
    private val context: Context,
    private val engine: Tn5000OrtEngine,
) {
    private val labelSnapshot = LabelSnapshot.load(context, "TN5000")

    fun evaluateTestSet(
        rootUri: Uri,
        options: EvaluationOptions,
        outputDir: File? = null,
        fileStem: String? = null,
        progressCallback: ((processed: Int, total: Int, currentId: String, roiBitmap: Bitmap?) -> Unit)? = null,
    ): EvaluationSummary {
        val root = DocumentFile.fromTreeUri(context, rootUri)
            ?: error("Cannot access selected TN5000 root folder.")

        val annotationsDir = root.findFileStrict("Annotations")
        val jpegDir = root.findFileStrict("JPEGImages")
        val imageSetsDir = root.findFileStrict("ImageSets")
        val mainDir = imageSetsDir.findFileStrict("Main")
        val testFile = mainDir.findFileStrict("test.txt")

        val jpegMap = jpegDir.listFiles().associateBy { it.name.orEmpty() }
        val xmlMap = annotationsDir.listFiles().associateBy { it.name.orEmpty() }
        val testIds = context.contentResolver.readText(testFile.uri)
            .lineSequence()
            .map { it.trim() }
            .filter { it.isNotEmpty() }
            .toList()

        require(testIds.isNotEmpty()) { "test.txt is empty." }

        val preparedMode = engine.prepareMode(options.mode)
        val rows = ArrayList<SamplePrediction>(testIds.size)
        var peakJavaHeap = currentJavaHeapMb()
        var peakNativeHeap = currentNativeHeapMb()
        val startedNs = System.nanoTime()

        testIds.forEachIndexed { index, sampleId ->
            val xmlDoc = xmlMap["$sampleId.xml"] ?: error("Missing annotation: $sampleId.xml")
            val imgDoc = jpegMap["$sampleId.jpg"]
                ?: jpegMap["$sampleId.jpeg"]
                ?: jpegMap["$sampleId.png"]
                ?: error("Missing image file for sample $sampleId")

            val sample = VocXmlParser.parseSample(context.contentResolver, xmlDoc.uri, sampleId)
            val groundTruth = labelSnapshot.labelFor(sampleId) ?: sample.label
            val originalBitmap = context.contentResolver.decodeBitmap(imgDoc.uri)
            val roiBitmap = BitmapUtils.cropRoiTrainingAligned(
                src = originalBitmap,
                xmin = sample.bbox.xmin,
                ymin = sample.bbox.ymin,
                xmax = sample.bbox.xmax,
                ymax = sample.bbox.ymax,
                expandRatio = options.expandRatio,
                squareCrop = options.squareCrop,
            )
            val result = engine.infer(
                bitmap = roiBitmap,
                options = InferenceOptions(
                    mode = options.mode,
                    threshold = options.threshold,
                    temperature = options.temperature,
                    inputPreparation = InputPreparation.DIRECT_RESIZE,
                )
            )
            rows += SamplePrediction(
                sampleId = sampleId,
                groundTruth = groundTruth,
                predicted = result.predictedIndex,
                probability0 = result.probabilities.getOrElse(0) { 0f }.toDouble(),
                probability1 = result.probabilities.getOrElse(1) { 0f }.toDouble(),
                preprocessMs = result.preprocessMs,
                inferenceMs = result.inferenceMs,
                totalMs = result.totalMs,
                bbox = sample.bbox,
            )

            peakJavaHeap = max(peakJavaHeap, currentJavaHeapMb())
            peakNativeHeap = max(peakNativeHeap, currentNativeHeapMb())

            if (index == testIds.lastIndex || index % 10 == 0) {
                progressCallback?.invoke(index + 1, testIds.size, sampleId, roiBitmap)
            }
        }

        val metrics = MetricsUtils.computeBinaryMetrics(
            labels = rows.map { it.groundTruth },
            probs1 = rows.map { it.probability1 },
            threshold = options.threshold,
        )
        val totalRuntimeSec = (System.nanoTime() - startedNs) / 1_000_000_000.0
        val csvFile = writeCsv(rows, outputDir = outputDir, fileStem = fileStem)
        val summary = EvaluationSummary(
            mode = options.mode,
            assetsUsed = preparedMode.assetNames,
            modelBytesTotal = preparedMode.totalModelBytes,
            modePrepareMs = preparedMode.modePrepareMs,
            total = metrics.total,
            correct = metrics.correct,
            accuracy = metrics.accuracy,
            balancedAccuracy = metrics.balancedAccuracy,
            precisionMacro = metrics.precisionMacro,
            recallMacro = metrics.recallMacro,
            f1Macro = metrics.f1Macro,
            auc = metrics.auc,
            recallBenign = metrics.recall0,
            recallMalignant = metrics.recall1,
            tn = metrics.tn,
            fp = metrics.fp,
            fn = metrics.fn,
            tp = metrics.tp,
            avgPreprocessMs = rows.map { it.preprocessMs }.averageOrZero(),
            avgInferenceMs = rows.map { it.inferenceMs }.averageOrZero(),
            avgTotalMs = rows.map { it.totalMs }.averageOrZero(),
            p50TotalMs = MetricsUtils.percentile(rows.map { it.totalMs }, 50.0),
            p95TotalMs = MetricsUtils.percentile(rows.map { it.totalMs }, 95.0),
            totalRuntimeSec = totalRuntimeSec,
            peakJavaHeapMb = peakJavaHeap,
            peakNativeHeapMb = peakNativeHeap,
            csvFile = csvFile,
            summaryFile = File(csvFile.parentFile, csvFile.nameWithoutExtension + "_summary.txt"),
        )
        writeSummary(summary, options)
        return summary
    }

    private fun writeCsv(rows: List<SamplePrediction>, outputDir: File?, fileStem: String?): File {
        val outDir = outputDir ?: File(context.getExternalFilesDir(null), "eval_reports")
        outDir.mkdirs()
        val stem = fileStem ?: "tn5000_test_eval_${System.currentTimeMillis()}"
        val outFile = File(outDir, "$stem.csv")
        FileWriter(outFile).use { writer ->
            writer.appendLine("sample_id,gt,pred,prob_0,prob_1,correct,preprocess_ms,inference_ms,total_ms,xmin,ymin,xmax,ymax")
            rows.forEach { row ->
                writer.appendLine(
                    listOf(
                        row.sampleId,
                        row.groundTruth.toString(),
                        row.predicted.toString(),
                        formatDouble(row.probability0),
                        formatDouble(row.probability1),
                        (row.groundTruth == row.predicted).toString(),
                        formatDouble(row.preprocessMs),
                        formatDouble(row.inferenceMs),
                        formatDouble(row.totalMs),
                        row.bbox.xmin.toString(),
                        row.bbox.ymin.toString(),
                        row.bbox.xmax.toString(),
                        row.bbox.ymax.toString(),
                    ).joinToString(",")
                )
            }
        }
        return outFile
    }

    private fun writeSummary(summary: EvaluationSummary, options: EvaluationOptions) {
        FileWriter(summary.summaryFile).use { writer ->
            writer.appendLine("TN5000 Android deployment summary")
            writer.appendLine("========================================")
            writer.appendLine("mode=${summary.mode.title}")
            writer.appendLine("assets_used=${summary.assetsUsed.joinToString(";")}")
            writer.appendLine("model_bytes_total=${summary.modelBytesTotal}")
            writer.appendLine("mode_prepare_ms=${formatDouble(summary.modePrepareMs)}")
            writer.appendLine("threshold=${formatDouble(options.threshold)}")
            writer.appendLine("temperature=${formatDouble(options.temperature)}")
            writer.appendLine("expand_ratio=${formatDouble(options.expandRatio.toDouble())}")
            writer.appendLine("square_crop=${options.squareCrop}")
            writer.appendLine("label_snapshot=${labelSnapshot.sourcePath.ifBlank { "XML labels" }}")
            writer.appendLine("label_snapshot_count=${labelSnapshot.labels.size}")
            writer.appendLine()
            writer.appendLine("total=${summary.total}")
            writer.appendLine("correct=${summary.correct}")
            writer.appendLine("accuracy=${formatDouble(summary.accuracy)}")
            writer.appendLine("balanced_accuracy=${formatDouble(summary.balancedAccuracy)}")
            writer.appendLine("precision_macro=${formatDouble(summary.precisionMacro)}")
            writer.appendLine("recall_macro=${formatDouble(summary.recallMacro)}")
            writer.appendLine("f1_macro=${formatDouble(summary.f1Macro)}")
            writer.appendLine("auc=${formatDouble(summary.auc)}")
            writer.appendLine("recall_benign=${formatDouble(summary.recallBenign)}")
            writer.appendLine("recall_malignant=${formatDouble(summary.recallMalignant)}")
            writer.appendLine("confusion=[[TN=${summary.tn},FP=${summary.fp}],[FN=${summary.fn},TP=${summary.tp}]]")
            writer.appendLine()
            writer.appendLine("avg_preprocess_ms=${formatDouble(summary.avgPreprocessMs)}")
            writer.appendLine("avg_inference_ms=${formatDouble(summary.avgInferenceMs)}")
            writer.appendLine("avg_total_ms=${formatDouble(summary.avgTotalMs)}")
            writer.appendLine("p50_total_ms=${formatDouble(summary.p50TotalMs)}")
            writer.appendLine("p95_total_ms=${formatDouble(summary.p95TotalMs)}")
            writer.appendLine("total_runtime_sec=${formatDouble(summary.totalRuntimeSec)}")
            writer.appendLine("peak_java_heap_mb=${formatDouble(summary.peakJavaHeapMb)}")
            writer.appendLine("peak_native_heap_mb=${formatDouble(summary.peakNativeHeapMb)}")
            writer.appendLine("csv=${summary.csvFile.absolutePath}")
        }
    }

    private fun currentJavaHeapMb(): Double {
        val runtime = Runtime.getRuntime()
        val usedBytes = runtime.totalMemory() - runtime.freeMemory()
        return usedBytes / (1024.0 * 1024.0)
    }

    private fun currentNativeHeapMb(): Double {
        return Debug.getNativeHeapAllocatedSize() / (1024.0 * 1024.0)
    }

    private fun formatDouble(value: Double): String = "%.6f".format(value)
}

private fun List<Double>.averageOrZero(): Double = if (isEmpty()) 0.0 else average()

private fun DocumentFile.findFileStrict(name: String): DocumentFile {
    return requireNotNull(findFile(name)) { "Missing entry under dataset root: $name" }
}

private fun ContentResolver.readText(uri: Uri): String {
    return openInputStream(uri).use { input ->
        requireNotNull(input) { "Cannot open $uri" }
        input.bufferedReader().readText()
    }
}

private fun ContentResolver.decodeBitmap(uri: Uri): Bitmap {
    return openInputStream(uri).use { input ->
        requireNotNull(input) { "Cannot open image: $uri" }
        BitmapFactory.decodeStream(input) ?: error("Failed to decode bitmap: $uri")
    }
}

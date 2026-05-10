package com.afriste.tn5000ortdemo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.os.Debug
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.max
import kotlin.math.sqrt

class FileTn5000DatasetEvaluator(
    private val context: Context,
    private val engine: Tn5000OrtEngine,
) {
    private val labelSnapshot = LabelSnapshot.load(context, "TN5000")

    fun evaluateTestSet(
        rootDir: File,
        options: EvaluationOptions,
        outputDir: File,
        fileStem: String,
        progressCallback: ((processed: Int, total: Int, currentId: String, roiBitmap: Bitmap?) -> Unit)? = null,
    ): EvaluationSummary {
        val annotationsDir = File(rootDir, "Annotations")
        val jpegDir = File(rootDir, "JPEGImages")
        val testFile = File(rootDir, "ImageSets/Main/test.txt")
        require(annotationsDir.isDirectory) { "Missing ${annotationsDir.absolutePath}" }
        require(jpegDir.isDirectory) { "Missing ${jpegDir.absolutePath}" }
        require(testFile.isFile) { "Missing ${testFile.absolutePath}" }

        val jpegMap = jpegDir.listFiles().orEmpty().associateBy { it.name }
        val xmlMap = annotationsDir.listFiles().orEmpty().associateBy { it.name }
        val testIds = testFile.readLines()
            .map { it.trim() }
            .filter { it.isNotEmpty() }
        require(testIds.isNotEmpty()) { "test.txt is empty." }

        val preparedMode = engine.prepareMode(options.mode)
        val rows = ArrayList<SamplePrediction>(testIds.size)
        var peakJavaHeap = currentJavaHeapMb()
        var peakNativeHeap = currentNativeHeapMb()
        val startedNs = System.nanoTime()

        testIds.forEachIndexed { index, sampleId ->
            val xmlFile = xmlMap["$sampleId.xml"] ?: error("Missing annotation: $sampleId.xml")
            val imageFile = jpegMap["$sampleId.jpg"]
                ?: jpegMap["$sampleId.jpeg"]
                ?: jpegMap["$sampleId.png"]
                ?: error("Missing image file for sample $sampleId")
            val sample = VocXmlParser.parseSample(xmlFile, sampleId)
            val groundTruth = labelSnapshot.labelFor(sampleId) ?: sample.label
            val originalBitmap = BitmapFactory.decodeFile(imageFile.absolutePath)
                ?: error("Failed to decode image: ${imageFile.absolutePath}")
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
        val csvFile = writeCsv(rows, outputDir, fileStem)
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

    private fun writeCsv(rows: List<SamplePrediction>, outputDir: File, fileStem: String): File {
        outputDir.mkdirs()
        val outFile = File(outputDir, "$fileStem.csv")
        FileWriter(outFile).use { writer ->
            writer.appendLine("sample_id,gt,pred,prob_0,prob_1,correct,preprocess_ms,inference_ms,total_ms,xmin,ymin,xmax,ymax")
            rows.forEach { row ->
                writer.appendLine(
                    listOf(
                        row.sampleId,
                        row.groundTruth.toString(),
                        row.predicted.toString(),
                        row.probability0.fmt(),
                        row.probability1.fmt(),
                        (row.groundTruth == row.predicted).toString(),
                        row.preprocessMs.fmt(),
                        row.inferenceMs.fmt(),
                        row.totalMs.fmt(),
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
            writer.appendLine("TN5000 Android file-mode deployment summary")
            writer.appendLine("mode=${summary.mode.title}")
            writer.appendLine("assets_used=${summary.assetsUsed.joinToString(";")}")
            writer.appendLine("model_bytes_total=${summary.modelBytesTotal}")
            writer.appendLine("mode_prepare_ms=${summary.modePrepareMs.fmt()}")
            writer.appendLine("threshold=${options.threshold.fmt()}")
            writer.appendLine("temperature=${options.temperature.fmt()}")
            writer.appendLine("expand_ratio=${options.expandRatio.toDouble().fmt()}")
            writer.appendLine("square_crop=${options.squareCrop}")
            writer.appendLine("label_snapshot=${labelSnapshot.sourcePath.ifBlank { "XML labels" }}")
            writer.appendLine("label_snapshot_count=${labelSnapshot.labels.size}")
            writer.appendLine("total=${summary.total}")
            writer.appendLine("accuracy=${summary.accuracy.fmt()}")
            writer.appendLine("balanced_accuracy=${summary.balancedAccuracy.fmt()}")
            writer.appendLine("f1_macro=${summary.f1Macro.fmt()}")
            writer.appendLine("auc=${summary.auc.fmt()}")
            writer.appendLine("avg_preprocess_ms=${summary.avgPreprocessMs.fmt()}")
            writer.appendLine("avg_inference_ms=${summary.avgInferenceMs.fmt()}")
            writer.appendLine("avg_total_ms=${summary.avgTotalMs.fmt()}")
            writer.appendLine("p50_total_ms=${summary.p50TotalMs.fmt()}")
            writer.appendLine("p95_total_ms=${summary.p95TotalMs.fmt()}")
            writer.appendLine("peak_java_heap_mb=${summary.peakJavaHeapMb.fmt()}")
            writer.appendLine("peak_native_heap_mb=${summary.peakNativeHeapMb.fmt()}")
            writer.appendLine("csv=${summary.csvFile.absolutePath}")
        }
    }

    private fun currentJavaHeapMb(): Double {
        val runtime = Runtime.getRuntime()
        return (runtime.totalMemory() - runtime.freeMemory()) / (1024.0 * 1024.0)
    }

    private fun currentNativeHeapMb(): Double {
        return Debug.getNativeHeapAllocatedSize() / (1024.0 * 1024.0)
    }
}

class FilePaperBatchRunner(
    private val context: Context,
    private val engine: Tn5000OrtEngine,
) {
    private val evaluator = FileTn5000DatasetEvaluator(context, engine)

    fun runPaperBatch(
        rootDir: File,
        baseOptions: EvaluationOptions,
        batchOptions: PaperBatchOptions,
        progressCallback: ((batchIndex: Int, batchTotal: Int, message: String, evalProcessed: Int, evalTotal: Int, currentId: String?) -> Unit)? = null,
    ): PaperBatchSummary {
        val batchDir = createBatchDir()
        val records = mutableListOf<PaperBatchRunRecord>()
        val totalRuns = batchOptions.selectedModes.sumOf { batchOptions.coldRunsPerMode + batchOptions.hotRunsPerMode }
        var completedRuns = 0
        writeDeviceInfo(batchDir, rootDir, batchOptions, baseOptions)

        batchOptions.selectedModes.forEach { mode ->
            repeat(batchOptions.coldRunsPerMode) { idx ->
                engine.clearSessionCache()
                System.gc()
                val phase = "cold"
                val runNumber = idx + 1
                val stem = "${mode.title.lowercase(Locale.US)}_${phase}${runNumber}".replace("/", "-")
                progressCallback?.invoke(completedRuns, totalRuns, "Preparing ${mode.title} $phase run $runNumber", 0, 0, null)
                val summary = evaluator.evaluateTestSet(
                    rootDir = rootDir,
                    options = baseOptions.copy(mode = mode),
                    outputDir = batchDir,
                    fileStem = stem,
                ) { processed, total, currentId, _ ->
                    progressCallback?.invoke(completedRuns, totalRuns, "Running ${mode.title} $phase run $runNumber", processed, total, currentId)
                }
                records += PaperBatchRunRecord(mode, phase, runNumber, summary)
                completedRuns += 1
            }
            repeat(batchOptions.hotRunsPerMode) { idx ->
                val phase = "hot"
                val runNumber = idx + 1
                val stem = "${mode.title.lowercase(Locale.US)}_${phase}${runNumber}".replace("/", "-")
                progressCallback?.invoke(completedRuns, totalRuns, "Preparing ${mode.title} $phase run $runNumber", 0, 0, null)
                val summary = evaluator.evaluateTestSet(
                    rootDir = rootDir,
                    options = baseOptions.copy(mode = mode),
                    outputDir = batchDir,
                    fileStem = stem,
                ) { processed, total, currentId, _ ->
                    progressCallback?.invoke(completedRuns, totalRuns, "Running ${mode.title} $phase run $runNumber", processed, total, currentId)
                }
                records += PaperBatchRunRecord(mode, phase, runNumber, summary)
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
        return PaperBatchSummary(batchDir, manifestCsv, aggregateModeCsv, aggregateModePhaseCsv, summaryTxt, deviceInfoTxt, totalRuns, completedRuns, records)
    }

    private fun createBatchDir(): File {
        val root = File(context.getExternalFilesDir(null), "eval_reports")
        root.mkdirs()
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        return File(root, "headless_file_batch_$stamp").apply { mkdirs() }
    }

    private fun writeDeviceInfo(batchDir: File, rootDir: File, batchOptions: PaperBatchOptions, baseOptions: EvaluationOptions) {
        FileWriter(File(batchDir, "device_info.txt")).use { writer ->
            writer.appendLine("TN5000 Android file-mode paper batch device info")
            writer.appendLine("manufacturer=${Build.MANUFACTURER}")
            writer.appendLine("brand=${Build.BRAND}")
            writer.appendLine("model=${Build.MODEL}")
            writer.appendLine("device=${Build.DEVICE}")
            writer.appendLine("product=${Build.PRODUCT}")
            writer.appendLine("sdk_int=${Build.VERSION.SDK_INT}")
            writer.appendLine("release=${Build.VERSION.RELEASE}")
            writer.appendLine("abis=${Build.SUPPORTED_ABIS.joinToString(";")}")
            writer.appendLine("dataset_root=${rootDir.absolutePath}")
            writer.appendLine("selected_modes=${batchOptions.selectedModes.joinToString(",") { it.title }}")
            writer.appendLine("cold_runs_per_mode=${batchOptions.coldRunsPerMode}")
            writer.appendLine("hot_runs_per_mode=${batchOptions.hotRunsPerMode}")
            writer.appendLine("threshold=${baseOptions.threshold.fmt()}")
            writer.appendLine("temperature=${baseOptions.temperature.fmt()}")
            writer.appendLine("expand_ratio=${baseOptions.expandRatio.toDouble().fmt()}")
            writer.appendLine("square_crop=${baseOptions.squareCrop}")
            writer.appendLine("note=${batchOptions.note}")
        }
    }

    private fun writeRunManifest(records: List<PaperBatchRunRecord>, outFile: File) {
        FileWriter(outFile).use { writer ->
            writer.appendLine("mode,phase,phase_index,total,correct,accuracy,balanced_accuracy,f1_macro,auc,avg_preprocess_ms,avg_inference_ms,avg_total_ms,p50_total_ms,p95_total_ms,total_runtime_sec,peak_java_heap_mb,peak_native_heap_mb,model_mb,mode_prepare_ms,csv_path,summary_path")
            records.forEach { r ->
                val s = r.summary
                writer.appendLine(
                    listOf(
                        r.mode.title,
                        r.phase,
                        r.phaseIndex.toString(),
                        s.total.toString(),
                        s.correct.toString(),
                        s.accuracy.fmt(),
                        s.balancedAccuracy.fmt(),
                        s.f1Macro.fmt(),
                        s.auc.fmt(),
                        s.avgPreprocessMs.fmt(),
                        s.avgInferenceMs.fmt(),
                        s.avgTotalMs.fmt(),
                        s.p50TotalMs.fmt(),
                        s.p95TotalMs.fmt(),
                        s.totalRuntimeSec.fmt(),
                        s.peakJavaHeapMb.fmt(),
                        s.peakNativeHeapMb.fmt(),
                        (s.modelBytesTotal / (1024.0 * 1024.0)).fmt(),
                        s.modePrepareMs.fmt(),
                        csvEscape(s.csvFile.absolutePath),
                        csvEscape(s.summaryFile.absolutePath),
                    ).joinToString(",")
                )
            }
        }
    }

    private fun writeAggregateByMode(records: List<PaperBatchRunRecord>, outFile: File) {
        FileWriter(outFile).use { writer ->
            writer.appendLine("mode,n_runs,accuracy_mean,accuracy_std,balanced_accuracy_mean,balanced_accuracy_std,f1_macro_mean,f1_macro_std,auc_mean,auc_std,avg_total_ms_mean,avg_total_ms_std,avg_inference_ms_mean,avg_inference_ms_std,p95_total_ms_mean,p95_total_ms_std,peak_java_heap_mb_mean,peak_native_heap_mb_mean")
            records.groupBy { it.mode }.entries.sortedBy { it.key.ordinal }.forEach { (mode, group) ->
                writer.appendLine(
                    listOf(
                        mode.title,
                        group.size.toString(),
                        meanOf(group) { it.summary.accuracy }.fmt(),
                        stdOf(group) { it.summary.accuracy }.fmt(),
                        meanOf(group) { it.summary.balancedAccuracy }.fmt(),
                        stdOf(group) { it.summary.balancedAccuracy }.fmt(),
                        meanOf(group) { it.summary.f1Macro }.fmt(),
                        stdOf(group) { it.summary.f1Macro }.fmt(),
                        meanOf(group) { it.summary.auc }.fmt(),
                        stdOf(group) { it.summary.auc }.fmt(),
                        meanOf(group) { it.summary.avgTotalMs }.fmt(),
                        stdOf(group) { it.summary.avgTotalMs }.fmt(),
                        meanOf(group) { it.summary.avgInferenceMs }.fmt(),
                        stdOf(group) { it.summary.avgInferenceMs }.fmt(),
                        meanOf(group) { it.summary.p95TotalMs }.fmt(),
                        stdOf(group) { it.summary.p95TotalMs }.fmt(),
                        meanOf(group) { it.summary.peakJavaHeapMb }.fmt(),
                        meanOf(group) { it.summary.peakNativeHeapMb }.fmt(),
                    ).joinToString(",")
                )
            }
        }
    }

    private fun writeAggregateByModePhase(records: List<PaperBatchRunRecord>, outFile: File) {
        FileWriter(outFile).use { writer ->
            writer.appendLine("mode,phase,n_runs,auc_mean,avg_total_ms_mean,avg_inference_ms_mean,p95_total_ms_mean,mode_prepare_ms_mean")
            records.groupBy { Pair(it.mode, it.phase) }.entries.sortedWith(compareBy({ it.key.first.ordinal }, { it.key.second })).forEach { (key, group) ->
                writer.appendLine(
                    listOf(
                        key.first.title,
                        key.second,
                        group.size.toString(),
                        meanOf(group) { it.summary.auc }.fmt(),
                        meanOf(group) { it.summary.avgTotalMs }.fmt(),
                        meanOf(group) { it.summary.avgInferenceMs }.fmt(),
                        meanOf(group) { it.summary.p95TotalMs }.fmt(),
                        meanOf(group) { it.summary.modePrepareMs }.fmt(),
                    ).joinToString(",")
                )
            }
        }
    }

    private fun writeBatchSummary(records: List<PaperBatchRunRecord>, batchOptions: PaperBatchOptions, baseOptions: EvaluationOptions, outFile: File) {
        FileWriter(outFile).use { writer ->
            writer.appendLine("TN5000 Android file-mode paper batch summary")
            writer.appendLine("selected_modes=${batchOptions.selectedModes.joinToString(",") { it.title }}")
            writer.appendLine("cold_runs_per_mode=${batchOptions.coldRunsPerMode}")
            writer.appendLine("hot_runs_per_mode=${batchOptions.hotRunsPerMode}")
            writer.appendLine("threshold=${baseOptions.threshold.fmt()}")
            writer.appendLine("temperature=${baseOptions.temperature.fmt()}")
            writer.appendLine("expand_ratio=${baseOptions.expandRatio.toDouble().fmt()}")
            writer.appendLine("square_crop=${baseOptions.squareCrop}")
            batchOptions.selectedModes.forEach { mode ->
                val group = records.filter { it.mode == mode }
                if (group.isEmpty()) return@forEach
                writer.appendLine("")
                writer.appendLine("[${mode.title}]")
                writer.appendLine("runs=${group.size}")
                writer.appendLine("auc_mean=${meanOf(group) { it.summary.auc }.fmt()}")
                writer.appendLine("avg_total_ms_mean=${meanOf(group) { it.summary.avgTotalMs }.fmt()}")
                writer.appendLine("avg_inference_ms_mean=${meanOf(group) { it.summary.avgInferenceMs }.fmt()}")
                writer.appendLine("p95_total_ms_mean=${meanOf(group) { it.summary.p95TotalMs }.fmt()}")
            }
        }
    }

    private fun meanOf(records: List<PaperBatchRunRecord>, selector: (PaperBatchRunRecord) -> Double): Double =
        if (records.isEmpty()) 0.0 else records.map(selector).average()

    private fun stdOf(records: List<PaperBatchRunRecord>, selector: (PaperBatchRunRecord) -> Double): Double {
        if (records.isEmpty()) return 0.0
        val values = records.map(selector)
        val mean = values.average()
        return sqrt(values.map { (it - mean) * (it - mean) }.average())
    }

    private fun csvEscape(value: String): String {
        val needsQuote = value.contains(',') || value.contains('"') || value.contains('\n')
        return if (needsQuote) "\"${value.replace("\"", "\"\"")}\"" else value
    }
}

private fun List<Double>.averageOrZero(): Double = if (isEmpty()) 0.0 else average()

private fun Double.fmt(): String = "%.6f".format(Locale.US, this)

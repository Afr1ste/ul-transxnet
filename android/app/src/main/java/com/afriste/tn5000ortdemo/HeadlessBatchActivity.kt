package com.afriste.tn5000ortdemo

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.documentfile.provider.DocumentFile
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class HeadlessBatchActivity : ComponentActivity() {
    private var engine: Tn5000OrtEngine? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Thread {
            runHeadlessBatch()
        }.start()
    }

    private fun runHeadlessBatch() {
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val statusDir = File(getExternalFilesDir(null), "headless_status").apply { mkdirs() }
        val statusFile = File(statusDir, "headless_batch_$stamp.status.json")
        val latestStatusFile = File(statusDir, "headless_batch_latest.status.json")
        fun writeStatus(state: String, stage: String, extra: Map<String, String> = emptyMap()) {
            val lines = mutableListOf(
                "state" to state,
                "stage" to stage,
                "updated_at" to SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US).format(Date()),
                "status_path" to statusFile.absolutePath,
                "latest_status_path" to latestStatusFile.absolutePath,
            )
            extra.forEach { (k, v) -> lines += k to v }
            val body = buildString {
                append("{\n")
                lines.forEachIndexed { idx, pair ->
                    append("  \"")
                    append(jsonEscape(pair.first))
                    append("\": \"")
                    append(jsonEscape(pair.second))
                    append("\"")
                    if (idx != lines.lastIndex) append(",")
                    append("\n")
                }
                append("}\n")
            }
            statusFile.writeText(body, Charsets.UTF_8)
            latestStatusFile.writeText(body, Charsets.UTF_8)
        }

        try {
            writeStatus("running", "init")
            val datasetUri = persistedTn5000Uri()
            val datasetPath = intent.getStringExtra("dataset_path")
                ?: File(getExternalFilesDir(null), "TN5000_forReview").absolutePath
            val datasetDir = File(datasetPath)
            val modes = parseModes()
            val coldRuns = intent.getIntExtra("cold_runs", 1).coerceAtLeast(0)
            val hotRuns = intent.getIntExtra("hot_runs", 2).coerceAtLeast(0)
            val threshold = intent.getDoubleExtra("threshold", 0.85)
            val temperature = intent.getDoubleExtra("temperature", 1.0)
            val expandRatio = intent.getFloatExtra("expand_ratio", 0.30f)
            val squareCrop = intent.getBooleanExtra("square_crop", false)
            val note = intent.getStringExtra("note").orEmpty()

            engine = Tn5000OrtEngine(this)
            val baseOptions = EvaluationOptions(
                mode = modes.first(),
                threshold = threshold,
                temperature = temperature,
                expandRatio = expandRatio,
                squareCrop = squareCrop,
            )
            val batchOptions = PaperBatchOptions(
                selectedModes = modes,
                coldRunsPerMode = coldRuns,
                hotRunsPerMode = hotRuns,
                note = note,
            )
            writeStatus(
                "running",
                "batch",
                mapOf(
                    "modes" to modes.joinToString(",") { it.name },
                    "cold_runs" to coldRuns.toString(),
                    "hot_runs" to hotRuns.toString(),
                    "dataset_uri" to datasetUri?.toString().orEmpty(),
                    "dataset_path" to datasetDir.absolutePath,
                )
            )
            val progress: (Int, Int, String, Int, Int, String?) -> Unit = { batchIndex, batchTotal, message, evalProcessed, evalTotal, currentId ->
                writeStatus(
                    "running",
                    "batch",
                    mapOf(
                        "batch_index" to batchIndex.toString(),
                        "batch_total" to batchTotal.toString(),
                        "message" to message,
                        "eval_processed" to evalProcessed.toString(),
                        "eval_total" to evalTotal.toString(),
                        "current_id" to currentId.orEmpty(),
                    )
                )
            }
            val summary = if (datasetUri != null) {
                PaperBatchRunner(this, requireNotNull(engine)).runPaperBatch(
                    rootUri = datasetUri,
                    baseOptions = baseOptions,
                    batchOptions = batchOptions,
                    progressCallback = progress,
                )
            } else {
                require(datasetDir.isDirectory) {
                    "No persisted TN5000_forReview tree permission and file dataset path does not exist: ${datasetDir.absolutePath}"
                }
                FilePaperBatchRunner(this, requireNotNull(engine)).runPaperBatch(
                    rootDir = datasetDir,
                    baseOptions = baseOptions,
                    batchOptions = batchOptions,
                    progressCallback = progress,
                )
            }
            writeStatus(
                "completed",
                "done",
                mapOf(
                    "batch_dir" to summary.batchDir.absolutePath,
                    "run_manifest_csv" to summary.runManifestCsv.absolutePath,
                    "aggregate_by_mode_csv" to summary.aggregateByModeCsv.absolutePath,
                    "aggregate_by_mode_phase_csv" to summary.aggregateByModePhaseCsv.absolutePath,
                    "summary_txt" to summary.summaryTxt.absolutePath,
                    "device_info_txt" to summary.deviceInfoTxt.absolutePath,
                    "finished_runs" to summary.finishedRuns.toString(),
                    "total_runs" to summary.totalRuns.toString(),
                )
            )
        } catch (e: Exception) {
            val errFile = File(statusDir, "headless_batch_$stamp.error.txt")
            FileWriter(errFile).use { it.write(e.stackTraceToString()) }
            writeStatus(
                "failed",
                "failed",
                mapOf(
                    "error" to e.toString(),
                    "error_trace" to errFile.absolutePath,
                )
            )
        } finally {
            engine?.close()
            finish()
        }
    }

    private fun persistedTn5000Uri(): Uri? {
        return contentResolver.persistedUriPermissions
            .filter { it.isReadPermission }
            .mapNotNull { permission ->
                val doc = DocumentFile.fromTreeUri(this, permission.uri)
                if (doc?.name == "TN5000_forReview") permission.uri else null
            }
            .firstOrNull()
    }

    private fun parseModes(): List<DeploymentMode> {
        val raw = intent.getStringExtra("modes").orEmpty()
        if (raw.isBlank()) {
            return listOf(
                DeploymentMode.MOBILE_RECOVERY_EFFFORMER_L1_KD,
                DeploymentMode.MOBILE_RECOVERY_EFFFORMER_L1_ECA_KD,
            )
        }
        return raw.split(',')
            .map { it.trim() }
            .filter { it.isNotEmpty() }
            .map { DeploymentMode.valueOf(it) }
    }

    private fun jsonEscape(value: String): String {
        return value
            .replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
    }
}

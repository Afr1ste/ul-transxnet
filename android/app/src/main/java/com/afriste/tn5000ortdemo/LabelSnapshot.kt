package com.afriste.tn5000ortdemo

import android.content.Context
import java.io.File

data class LabelSnapshot(
    val sourcePath: String,
    val labels: Map<String, Int>,
) {
    fun labelFor(imageId: String): Int? {
        val key = imageId.trim()
        if (key.isEmpty()) return null
        return labels[key] ?: labels[File(key).nameWithoutExtension]
    }

    companion object {
        fun load(context: Context, dataset: String): LabelSnapshot {
            val external = context.getExternalFilesDir(null)
            val candidates = listOfNotNull(
                File(context.filesDir, "paper_log_case_labels.csv"),
                File(context.filesDir, "label_snapshots/paper_log_case_labels.csv"),
                external?.let { File(it, "paper_log_case_labels.csv") },
                external?.let { File(it, "label_snapshots/paper_log_case_labels.csv") },
            )
            val file = candidates.firstOrNull { it.isFile } ?: return LabelSnapshot("", emptyMap())
            val labels = mutableMapOf<String, Int>()
            file.bufferedReader(Charsets.UTF_8).useLines { lines ->
                val iterator = lines.iterator()
                if (!iterator.hasNext()) return@useLines
                val header = parseCsvLine(iterator.next())
                val datasetIdx = header.indexOf("dataset")
                val imageIdx = header.indexOf("image_id")
                val labelIdx = header.indexOf("true_label")
                if (datasetIdx < 0 || imageIdx < 0 || labelIdx < 0) {
                    return@useLines
                }
                val datasetKey = dataset.uppercase()
                iterator.forEachRemaining { line ->
                    val cols = parseCsvLine(line)
                    if (cols.size <= maxOf(datasetIdx, imageIdx, labelIdx)) return@forEachRemaining
                    if (cols[datasetIdx].trim().uppercase() != datasetKey) return@forEachRemaining
                    val imageId = cols[imageIdx].trim()
                    val label = cols[labelIdx].trim().toDoubleOrNull()?.toInt() ?: return@forEachRemaining
                    if (imageId.isNotEmpty()) {
                        labels[imageId] = label
                        labels[File(imageId).nameWithoutExtension] = label
                    }
                }
            }
            return LabelSnapshot(file.absolutePath, labels)
        }

        private fun parseCsvLine(line: String): List<String> {
            val out = mutableListOf<String>()
            val current = StringBuilder()
            var inQuotes = false
            var i = 0
            while (i < line.length) {
                val c = line[i]
                when {
                    c == '"' && inQuotes && i + 1 < line.length && line[i + 1] == '"' -> {
                        current.append('"')
                        i += 1
                    }
                    c == '"' -> inQuotes = !inQuotes
                    c == ',' && !inQuotes -> {
                        out += current.toString()
                        current.setLength(0)
                    }
                    else -> current.append(c)
                }
                i += 1
            }
            out += current.toString()
            return out
        }
    }
}

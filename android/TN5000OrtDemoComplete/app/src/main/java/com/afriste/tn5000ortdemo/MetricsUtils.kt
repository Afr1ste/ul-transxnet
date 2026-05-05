package com.afriste.tn5000ortdemo

import kotlin.math.abs

private fun safeDiv(numerator: Int, denominator: Int): Double {
    return if (denominator == 0) 0.0 else numerator.toDouble() / denominator.toDouble()
}

data class BinaryMetrics(
    val total: Int,
    val correct: Int,
    val accuracy: Double,
    val balancedAccuracy: Double,
    val precisionMacro: Double,
    val recallMacro: Double,
    val f1Macro: Double,
    val auc: Double,
    val tn: Int,
    val fp: Int,
    val fn: Int,
    val tp: Int,
    val recall0: Double,
    val recall1: Double,
    val precision0: Double,
    val precision1: Double,
    val f10: Double,
    val f11: Double,
)

object MetricsUtils {
    fun computeBinaryMetrics(labels: List<Int>, probs1: List<Double>, threshold: Double): BinaryMetrics {
        require(labels.size == probs1.size) { "labels/probs size mismatch" }

        var tn = 0
        var fp = 0
        var fn = 0
        var tp = 0
        var correct = 0

        labels.indices.forEach { index ->
            val gt = labels[index]
            val pred = if (probs1[index] >= threshold) 1 else 0
            if (gt == pred) correct += 1
            when {
                gt == 0 && pred == 0 -> tn += 1
                gt == 0 && pred == 1 -> fp += 1
                gt == 1 && pred == 0 -> fn += 1
                gt == 1 && pred == 1 -> tp += 1
            }
        }

        val recall0 = safeDiv(tn, tn + fp)
        val recall1 = safeDiv(tp, tp + fn)
        val precision0 = safeDiv(tn, tn + fn)
        val precision1 = safeDiv(tp, tp + fp)
        val f10 = if (precision0 + recall0 == 0.0) 0.0 else 2.0 * precision0 * recall0 / (precision0 + recall0)
        val f11 = if (precision1 + recall1 == 0.0) 0.0 else 2.0 * precision1 * recall1 / (precision1 + recall1)
        val accuracy = safeDiv(correct, labels.size)
        val balancedAccuracy = (recall0 + recall1) / 2.0
        val precisionMacro = (precision0 + precision1) / 2.0
        val recallMacro = balancedAccuracy
        val f1Macro = (f10 + f11) / 2.0
        val auc = binaryAuc(labels, probs1)

        return BinaryMetrics(
            total = labels.size,
            correct = correct,
            accuracy = accuracy,
            balancedAccuracy = balancedAccuracy,
            precisionMacro = precisionMacro,
            recallMacro = recallMacro,
            f1Macro = f1Macro,
            auc = auc,
            tn = tn,
            fp = fp,
            fn = fn,
            tp = tp,
            recall0 = recall0,
            recall1 = recall1,
            precision0 = precision0,
            precision1 = precision1,
            f10 = f10,
            f11 = f11,
        )
    }

    fun percentile(values: List<Double>, percentile: Double): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        val p = percentile.coerceIn(0.0, 100.0)
        val rank = (p / 100.0) * (sorted.lastIndex)
        val lower = rank.toInt()
        val upper = kotlin.math.ceil(rank).toInt().coerceAtMost(sorted.lastIndex)
        if (lower == upper) return sorted[lower]
        val weight = rank - lower
        return sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }

    private fun binaryAuc(labels: List<Int>, scores: List<Double>): Double {
        if (labels.isEmpty() || labels.size != scores.size) return Double.NaN
        val paired = labels.zip(scores).sortedBy { it.second }
        val posCount = labels.count { it == 1 }
        val negCount = labels.count { it == 0 }
        if (posCount == 0 || negCount == 0) return Double.NaN

        var rank = 1.0
        var i = 0
        var sumPositiveRanks = 0.0
        while (i < paired.size) {
            var j = i + 1
            while (j < paired.size && abs(paired[j].second - paired[i].second) < 1e-12) {
                j += 1
            }
            val avgRank = (rank + (rank + (j - i) - 1.0)) / 2.0
            for (k in i until j) {
                if (paired[k].first == 1) {
                    sumPositiveRanks += avgRank
                }
            }
            rank += (j - i)
            i = j
        }
        return (sumPositiveRanks - posCount * (posCount + 1) / 2.0) / (posCount.toDouble() * negCount.toDouble())
    }
}

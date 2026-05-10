package com.afriste.tn5000ortdemo

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import java.nio.FloatBuffer
import java.util.EnumSet
import kotlin.math.exp
import kotlin.system.measureNanoTime

class Tn5000OrtEngine(
    private val context: Context,
) : AutoCloseable {

    private data class LoadedSession(
        val cacheKey: String,
        val assetName: String,
        val runtimeProfile: OrtRuntimeProfile,
        val session: OrtSession,
        val sizeBytes: Long,
        val initialLoadMs: Double,
    )

    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val sessionCache = linkedMapOf<String, LoadedSession>()
    private val labels = arrayOf("benign (0)", "malignant (1)")

    val defaultInputSize: Int = 256

    fun availableAssetNames(): List<String> = context.assets.list("")?.toList().orEmpty()

    fun summarizeAvailability(): String {
        val assets = availableAssetNames().toSet()
        return DeploymentMode.entries.joinToString(separator = "\n") { mode ->
            val missing = mode.assetNames.filterNot { assets.contains(it) }
            if (missing.isEmpty()) {
                "${mode.title}: ready (${mode.runtimeProfile.label}; ${mode.assetNames.joinToString()})"
            } else {
                "${mode.title}: missing ${missing.joinToString()}"
            }
        }
    }

    fun prepareMode(mode: DeploymentMode): PreparedMode {
        val assetsPresent = availableAssetNames().toSet()
        val missing = mode.assetNames.filterNot { it in assetsPresent }
        require(missing.isEmpty()) {
            "Mode ${mode.title} is missing assets: ${missing.joinToString()}"
        }

        val stats = mode.assetNames.map { assetName ->
            val cacheKey = sessionCacheKey(assetName, mode.runtimeProfile)
            val cached = sessionCache[cacheKey]
            if (cached != null) {
                ModelAssetStat(
                    assetName = assetName,
                    sizeBytes = cached.sizeBytes,
                    loadMs = 0.0,
                    newlyLoaded = false,
                )
            } else {
                loadSession(assetName, mode.runtimeProfile)
            }
        }
        return PreparedMode(mode = mode, assets = stats)
    }

    fun clearSessionCache() {
        sessionCache.values.forEach { it.session.close() }
        sessionCache.clear()
    }

    fun infer(
        bitmap: Bitmap,
        options: InferenceOptions,
    ): InferenceResult {
        val preparedMode = prepareMode(options.mode)
        val inputSize = options.mode.inputSize

        var originalBitmap: Bitmap? = null
        var flippedBitmap: Bitmap? = null
        lateinit var originalChw: FloatArray
        var flippedChw: FloatArray? = null

        val preprocessNs = measureNanoTime {
            originalBitmap = when (options.inputPreparation) {
                InputPreparation.CENTER_CROP -> BitmapUtils.centerCropAndResize(bitmap, inputSize, inputSize)
                InputPreparation.DIRECT_RESIZE -> BitmapUtils.resizeDirect(bitmap, inputSize, inputSize)
            }
            originalChw = preprocessToNchw(requireNotNull(originalBitmap))
            if (options.mode.useHflipTta) {
                flippedBitmap = BitmapUtils.horizontalFlip(requireNotNull(originalBitmap))
                flippedChw = preprocessToNchw(requireNotNull(flippedBitmap))
            }
        }

        val accumulated = FloatArray(2)
        var passCount = 0
        val inferenceNs = measureNanoTime {
            preparedMode.assetNames.forEach { assetName ->
                val session = requireNotNull(sessionCache[sessionCacheKey(assetName, options.mode.runtimeProfile)])
                addLogits(accumulated, runSession(session.session, originalChw, inputSize))
                passCount += 1
                if (flippedChw != null) {
                    addLogits(accumulated, runSession(session.session, flippedChw!!, inputSize))
                    passCount += 1
                }
            }
        }

        accumulated.indices.forEach { accumulated[it] /= passCount.toFloat().coerceAtLeast(1f) }
        val scaledLogits = accumulated.map { (it / options.temperature).toFloat() }.toFloatArray()
        val probabilities = softmax(scaledLogits)
        val malignantProb = probabilities.getOrElse(1) { 0f }
        val predictedIndex = if (malignantProb >= options.threshold) 1 else 0

        return InferenceResult(
            logits = scaledLogits,
            probabilities = probabilities,
            predictedIndex = predictedIndex,
            predictedLabel = labels[predictedIndex],
            preprocessMs = preprocessNs / 1_000_000.0,
            inferenceMs = inferenceNs / 1_000_000.0,
            totalMs = (preprocessNs + inferenceNs) / 1_000_000.0,
            threshold = options.threshold,
            temperature = options.temperature,
            usedAssets = preparedMode.assetNames,
        )
    }

    private fun loadSession(assetName: String, runtimeProfile: OrtRuntimeProfile): ModelAssetStat {
        val bytes = context.assets.open(assetName).use { it.readBytes() }
        var session: OrtSession? = null
        val elapsedNs = measureNanoTime {
            createSessionOptions(runtimeProfile).use { sessionOptions ->
                session = environment.createSession(bytes, sessionOptions)
            }
        }
        val loaded = LoadedSession(
            cacheKey = sessionCacheKey(assetName, runtimeProfile),
            assetName = assetName,
            runtimeProfile = runtimeProfile,
            session = requireNotNull(session),
            sizeBytes = bytes.size.toLong(),
            initialLoadMs = elapsedNs / 1_000_000.0,
        )
        sessionCache[loaded.cacheKey] = loaded
        return ModelAssetStat(
            assetName = assetName,
            sizeBytes = loaded.sizeBytes,
            loadMs = loaded.initialLoadMs,
            newlyLoaded = true,
        )
    }

    private fun sessionCacheKey(assetName: String, runtimeProfile: OrtRuntimeProfile): String {
        return "${runtimeProfile.name}::$assetName"
    }

    private fun createSessionOptions(runtimeProfile: OrtRuntimeProfile): OrtSession.SessionOptions {
        return OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            runtimeProfile.intraOpThreads?.let { setIntraOpNumThreads(it) }
            runtimeProfile.interOpThreads?.let { setInterOpNumThreads(it) }
            if (runtimeProfile.useXnnpack) {
                val threads = runtimeProfile.intraOpThreads?.toString() ?: "4"
                addXnnpack(mapOf("intra_op_num_threads" to threads))
            }
            if (runtimeProfile.useNnapi) {
                val flags = EnumSet.noneOf(NNAPIFlags::class.java)
                if (runtimeProfile.nnapiUseFp16) flags.add(NNAPIFlags.USE_FP16)
                if (runtimeProfile.nnapiUseNchw) flags.add(NNAPIFlags.USE_NCHW)
                addNnapi(flags)
            }
        }
    }

    private fun runSession(session: OrtSession, chw: FloatArray, inputSize: Int): FloatArray {
        val inputShape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        val inputTensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(chw), inputShape)
        inputTensor.use { tensor ->
            session.run(mapOf("image" to tensor)).use { outputs ->
                @Suppress("UNCHECKED_CAST")
                val logits = outputs[0].value as Array<FloatArray>
                return logits[0].copyOf()
            }
        }
    }

    private fun preprocessToNchw(bitmap: Bitmap): FloatArray {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        val planeSize = bitmap.width * bitmap.height
        val chw = FloatArray(3 * planeSize)

        pixels.indices.forEach { index ->
            val color = pixels[index]
            val r = ((color shr 16) and 0xFF) / 255.0f
            val g = ((color shr 8) and 0xFF) / 255.0f
            val b = (color and 0xFF) / 255.0f
            chw[index] = (r - mean[0]) / std[0]
            chw[planeSize + index] = (g - mean[1]) / std[1]
            chw[planeSize * 2 + index] = (b - mean[2]) / std[2]
        }
        return chw
    }

    private fun addLogits(target: FloatArray, src: FloatArray) {
        target.indices.forEach { target[it] += src.getOrElse(it) { 0f } }
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val maxValue = logits.maxOrNull() ?: 0f
        val exps = FloatArray(logits.size)
        var sum = 0.0
        logits.indices.forEach { index ->
            val e = exp((logits[index] - maxValue).toDouble())
            exps[index] = e.toFloat()
            sum += e
        }
        exps.indices.forEach { index -> exps[index] = (exps[index] / sum).toFloat() }
        return exps
    }

    override fun close() {
        sessionCache.values.forEach { it.session.close() }
        sessionCache.clear()
    }
}

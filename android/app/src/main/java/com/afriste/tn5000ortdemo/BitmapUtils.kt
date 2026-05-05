package com.afriste.tn5000ortdemo

import android.graphics.Bitmap
import android.graphics.Matrix
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

object BitmapUtils {
    fun centerCropAndResize(src: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        val scale = max(
            targetWidth.toFloat() / src.width.toFloat(),
            targetHeight.toFloat() / src.height.toFloat(),
        )
        val scaledWidth = (src.width * scale).roundToInt().coerceAtLeast(targetWidth)
        val scaledHeight = (src.height * scale).roundToInt().coerceAtLeast(targetHeight)
        val scaled = Bitmap.createScaledBitmap(src, scaledWidth, scaledHeight, true)

        val left = ((scaledWidth - targetWidth) / 2).coerceAtLeast(0)
        val top = ((scaledHeight - targetHeight) / 2).coerceAtLeast(0)
        return Bitmap.createBitmap(scaled, left, top, targetWidth, targetHeight)
    }

    fun resizeDirect(src: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        return Bitmap.createScaledBitmap(src, targetWidth, targetHeight, true)
    }

    fun horizontalFlip(src: Bitmap): Bitmap {
        val matrix = Matrix().apply { preScale(-1f, 1f) }
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height, matrix, true)
    }

    fun cropRoiTrainingAligned(
        src: Bitmap,
        xmin: Int,
        ymin: Int,
        xmax: Int,
        ymax: Int,
        expandRatio: Float,
        squareCrop: Boolean,
        minCropSize: Int = 64,
    ): Bitmap {
        require(xmax > xmin && ymax > ymin) { "Invalid bbox: ($xmin,$ymin,$xmax,$ymax)" }

        val w = src.width
        val h = src.height

        val bw = max((xmax - xmin).toFloat(), minCropSize.toFloat())
        val bh = max((ymax - ymin).toFloat(), minCropSize.toFloat())
        val cx = (xmin + xmax) / 2f
        val cy = (ymin + ymax) / 2f

        var cropW = bw * (1f + 2f * expandRatio)
        var cropH = bh * (1f + 2f * expandRatio)

        if (squareCrop) {
            val side = max(cropW, cropH)
            cropW = side
            cropH = side
        }

        var left = (cx - cropW / 2f).roundToInt()
        var top = (cy - cropH / 2f).roundToInt()
        var right = (cx + cropW / 2f).roundToInt()
        var bottom = (cy + cropH / 2f).roundToInt()

        left = left.coerceAtLeast(0)
        top = top.coerceAtLeast(0)
        right = right.coerceAtMost(w)
        bottom = bottom.coerceAtMost(h)

        if (squareCrop) {
            val currentW = right - left
            val currentH = bottom - top
            val side = min(max(currentW, currentH), min(w, h)).coerceAtLeast(1)
            val centerX = (left + right) / 2f
            val centerY = (top + bottom) / 2f
            left = (centerX - side / 2f).roundToInt().coerceIn(0, w - side)
            top = (centerY - side / 2f).roundToInt().coerceIn(0, h - side)
            right = left + side
            bottom = top + side
        }

        val finalW = (right - left).coerceAtLeast(1)
        val finalH = (bottom - top).coerceAtLeast(1)
        return Bitmap.createBitmap(src, left, top, finalW, finalH)
    }
}

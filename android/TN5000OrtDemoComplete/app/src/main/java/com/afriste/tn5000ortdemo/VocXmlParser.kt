package com.afriste.tn5000ortdemo

import android.content.ContentResolver
import android.net.Uri
import org.xmlpull.v1.XmlPullParser
import org.xmlpull.v1.XmlPullParserFactory

data class VocBBox(
    val xmin: Int,
    val ymin: Int,
    val xmax: Int,
    val ymax: Int,
)

data class VocParsedSample(
    val sampleId: String,
    val label: Int,
    val bbox: VocBBox,
)

object VocXmlParser {
    fun parseSample(resolver: ContentResolver, xmlUri: Uri, sampleId: String): VocParsedSample {
        val parser = XmlPullParserFactory.newInstance().newPullParser()
        resolver.openInputStream(xmlUri).use { input ->
            requireNotNull(input) { "Cannot open XML: $xmlUri" }
            parser.setInput(input, null)

            var eventType = parser.eventType
            var currentTag: String? = null

            var currentLabel: Int? = null
            var xmin: Int? = null
            var ymin: Int? = null
            var xmax: Int? = null
            var ymax: Int? = null

            val labels = mutableListOf<Int>()
            val boxes = mutableListOf<VocBBox>()

            while (eventType != XmlPullParser.END_DOCUMENT) {
                when (eventType) {
                    XmlPullParser.START_TAG -> {
                        currentTag = parser.name
                        if (currentTag == "object") {
                            currentLabel = null
                            xmin = null
                            ymin = null
                            xmax = null
                            ymax = null
                        }
                    }
                    XmlPullParser.TEXT -> {
                        val text = parser.text?.trim().orEmpty()
                        when (currentTag) {
                            "name" -> if (text.isNotEmpty() && currentLabel == null) currentLabel = text.toIntOrNull()
                            "xmin" -> xmin = text.toFloatOrNull()?.toInt()
                            "ymin" -> ymin = text.toFloatOrNull()?.toInt()
                            "xmax" -> xmax = text.toFloatOrNull()?.toInt()
                            "ymax" -> ymax = text.toFloatOrNull()?.toInt()
                        }
                    }
                    XmlPullParser.END_TAG -> {
                        if (parser.name == "object") {
                            val box = if (xmin != null && ymin != null && xmax != null && ymax != null && xmax!! > xmin!! && ymax!! > ymin!!) {
                                VocBBox(xmin!!, ymin!!, xmax!!, ymax!!)
                            } else null
                            if (box != null) {
                                boxes += box
                                labels += (currentLabel ?: labels.firstOrNull() ?: 0)
                            }
                        }
                        currentTag = null
                    }
                }
                eventType = parser.next()
            }

            require(boxes.isNotEmpty()) { "No valid bbox found in $sampleId" }
            val unionBox = VocBBox(
                xmin = boxes.minOf { it.xmin },
                ymin = boxes.minOf { it.ymin },
                xmax = boxes.maxOf { it.xmax },
                ymax = boxes.maxOf { it.ymax },
            )
            val label = labels.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key ?: labels.first()
            return VocParsedSample(sampleId = sampleId, label = label, bbox = unionBox)
        }
    }
}

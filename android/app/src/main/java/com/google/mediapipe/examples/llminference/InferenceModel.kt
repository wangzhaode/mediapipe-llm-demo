package com.google.mediapipe.examples.llminference

import android.content.Context
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import java.io.File
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow

class InferenceModel private constructor(context: Context) {
    private var llmInference: LlmInference
    private var startTime: Long = 0
    private var firstToken: Boolean = true
    private var prefillTokens: Int = 0
    private var decodeTokens: Int = 0
    private var prefillSpeed: Double = 0.0
    private val modelExists: Boolean
        get() = File(MODEL_PATH).exists()

    private val _partialResults = MutableSharedFlow<Pair<String, Boolean>>(
        extraBufferCapacity = 1,
        onBufferOverflow = BufferOverflow.DROP_OLDEST
    )
    val partialResults: SharedFlow<Pair<String, Boolean>> = _partialResults.asSharedFlow()

    init {
        if (!modelExists) {
            throw IllegalArgumentException("Model not found at path: $MODEL_PATH")
        }

        val options = LlmInference.LlmInferenceOptions.builder()
            .setModelPath(MODEL_PATH)
            .setMaxTokens(1024)
            .setResultListener { partialResult, done ->
                if (firstToken) {
                    val duration = System.nanoTime() - startTime
                    val prefillS = duration / 1_000_000_000.0
                    prefillSpeed = prefillTokens / prefillS
                    firstToken = false
                    startTime = System.nanoTime()
                } else {
                    decodeTokens += 1
                }
                if (done) {
                    val duration = System.nanoTime() - startTime
                    val decodeS = duration / 1_000_000_000.0
                    val decodeSpeed = decodeTokens / decodeS
                    val resultSummary = buildString {
                        appendLine(partialResult)
                        appendLine("\n-----------------------\n")
                        appendLine("prefill speed: ${"%.2f".format(prefillSpeed)} token/s for $prefillTokens tokens")
                        appendLine("decode  speed: ${"%.2f".format(decodeSpeed)} token/s for $decodeTokens tokens")
                    }
                    _partialResults.tryEmit(resultSummary to done)
                    decodeTokens = 0
                } else {
                    _partialResults.tryEmit(partialResult to done)
                }
            }
            .build()

        llmInference = LlmInference.createFromOptions(context, options)
    }

    fun generateResponseAsync(prompt: String) {
        // Add the gemma prompt prefix to trigger the response.
        val gemmaPrompt = prompt + "<start_of_turn>model\n"
        prefillTokens = llmInference.sizeInTokens(gemmaPrompt)
        startTime = System.nanoTime()
        firstToken = true
        llmInference.generateResponseAsync(gemmaPrompt)
    }

    companion object {
        // NB: Make sure the filename is *unique* per model you use!
        // Weight caching is currently based on filename alone.
        private const val MODEL_PATH = "/data/local/tmp/llm/model.bin"
        private var instance: InferenceModel? = null

        fun getInstance(context: Context): InferenceModel {
            return if (instance != null) {
                instance!!
            } else {
                InferenceModel(context).also { instance = it }
            }
        }
    }
}

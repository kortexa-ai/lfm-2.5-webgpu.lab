import {
  AutoTokenizer,
  AutoModelForCausalLM,
  TextStreamer,
} from "@huggingface/transformers";

let tokenizer = null;
let model = null;

const MODEL_ID = "LiquidAI/LFM2.5-1.2B-Thinking-ONNX";

// Check if model files are already in the browser cache
async function isModelCached(modelId) {
  try {
    const cache = await caches.open("transformers-cache");
    const keys = await cache.keys();
    return keys.some((req) => req.url.includes(modelId.replace("/", "%2F")) || req.url.includes(modelId));
  } catch {
    return false;
  }
}

self.onmessage = async (e) => {
  const { type, data } = e.data;

  if (type === "check") {
    const cached = await isModelCached(data?.modelId || MODEL_ID);
    self.postMessage({ type: "cache_status", data: { cached } });
    return;
  }

  if (type === "load") {
    try {
      self.postMessage({ type: "status", data: "Loading tokenizer..." });

      tokenizer = await AutoTokenizer.from_pretrained(data?.modelId || MODEL_ID, {
        progress_callback: (progress) => {
          self.postMessage({ type: "progress", data: progress });
        },
      });

      self.postMessage({ type: "status", data: "Loading model with WebGPU..." });

      model = await AutoModelForCausalLM.from_pretrained(data?.modelId || MODEL_ID, {
        device: "webgpu",
        dtype: data?.dtype || "q4f16",
        progress_callback: (progress) => {
          self.postMessage({ type: "progress", data: progress });
        },
      });

      self.postMessage({ type: "loaded" });
    } catch (err) {
      self.postMessage({ type: "error", data: err.message });
    }
  }

  if (type === "generate") {
    if (!tokenizer || !model) {
      self.postMessage({ type: "error", data: "Model not loaded yet" });
      return;
    }

    try {
      const messages = data.messages || [];

      // Apply chat template
      const inputs = tokenizer.apply_chat_template(messages, {
        add_generation_prompt: true,
        return_dict: true,
      });

      const promptTokens = inputs.input_ids.dims[1];
      self.postMessage({ type: "generate_start", data: { promptTokens } });

      const startTime = performance.now();
      let tokenCount = 0;

      // Stream tokens back to main thread
      const streamer = new TextStreamer(tokenizer, {
        skip_prompt: true,
        callback_function: (text) => {
          tokenCount++;
          const elapsed = (performance.now() - startTime) / 1000;
          self.postMessage({
            type: "token",
            data: {
              text,
              tokenCount,
              tokensPerSec: tokenCount / elapsed,
            },
          });
        },
      });

      await model.generate({
        ...inputs,
        max_new_tokens: data.maxTokens || 512,
        temperature: data.temperature ?? 0.7,
        do_sample: data.temperature > 0,
        streamer,
      });

      const elapsed = (performance.now() - startTime) / 1000;
      self.postMessage({
        type: "generate_done",
        data: {
          tokenCount,
          elapsed,
          tokensPerSec: tokenCount / elapsed,
        },
      });
    } catch (err) {
      self.postMessage({ type: "error", data: err.message });
    }
  }
};

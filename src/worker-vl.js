import * as ort from "onnxruntime-web/webgpu";
import { AutoTokenizer } from "@huggingface/transformers";

const MODEL_ID = "LiquidAI/LFM2.5-VL-1.6B-ONNX";
const MODEL_BASE = `https://huggingface.co/${MODEL_ID}/resolve/main/onnx`;

// Architecture constants (from config.json → text_config)
const HIDDEN_SIZE = 2048;
const NUM_KV_HEADS = 8;
const HEAD_DIM = 64; // 2048 / 32
const CONV_CACHE_LEN = 3;

let tokenizer = null;
let embedTokens = null;
let decoder = null;

ort.env.wasm.numThreads = 1;

const CACHE_NAME = "onnx-model-cache";

// Fetch a URL with Cache API caching and progress reporting
// Uses stable URL (no query params) as cache key since HF signed URLs change each time
async function cachedFetch(url, label) {
  const stableUrl = url.split("?")[0];
  const cache = await caches.open(CACHE_NAME);

  // Check cache first
  const cached = await cache.match(stableUrl);
  if (cached) {
    console.log("[worker-vl] cache hit:", stableUrl);
    return cached.arrayBuffer();
  }

  // Fetch with progress
  console.log("[worker-vl] downloading:", stableUrl);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);

  const total = parseInt(response.headers.get("content-length") || "0", 10);
  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    if (total > 0) {
      self.postMessage({
        type: "progress",
        data: { status: "progress", file: label, loaded, total },
      });
    }
  }

  // Combine chunks into a single ArrayBuffer
  const buf = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    buf.set(chunk, offset);
    offset += chunk.byteLength;
  }

  // Store in cache with stable URL key
  await cache.put(new Request(stableUrl), new Response(buf.buffer.slice(0), {
    headers: { "Content-Type": "application/octet-stream", "Content-Length": String(loaded) },
  }));

  return buf.buffer;
}

// Check if all model files are in the cache
async function checkModelCached() {
  try {
    const cache = await caches.open(CACHE_NAME);
    // Check for the main decoder file as a proxy for "model is cached"
    const key = `${MODEL_BASE}/decoder_q4.onnx`.split("?")[0];
    const hit = await cache.match(key);
    return !!hit;
  } catch {
    return false;
  }
}

// Load ONNX session with external data files, using Cache API
async function loadSession(name, dataFiles = 1, label) {
  const onnxUrl = `${MODEL_BASE}/${name}.onnx`;
  const onnxBuf = await cachedFetch(onnxUrl, label || `${name}.onnx`);

  const externalData = [];
  for (let i = 0; i < dataFiles; i++) {
    const suffix = i === 0 ? "" : `_${i}`;
    const fileName = `${name}.onnx_data${suffix}`;
    const dataUrl = `${MODEL_BASE}/${fileName}`;
    const dataBuf = await cachedFetch(dataUrl, label ? `${label} data` : fileName);
    externalData.push({ path: fileName, data: new Uint8Array(dataBuf) });
  }

  return ort.InferenceSession.create(new Uint8Array(onnxBuf), {
    executionProviders: ["webgpu"],
    externalData,
  });
}

// Get text embeddings via ONNX session
async function getTextEmbeddings(ids) {
  const tensor = new ort.Tensor(
    "int64",
    new BigInt64Array(ids.map(BigInt)),
    [1, ids.length]
  );
  const out = await embedTokens.run({ input_ids: tensor });
  return out.inputs_embeds;
}

// Initialize KV cache — dynamically from decoder input names
function initCache() {
  const cache = {};
  for (const name of decoder.inputNames) {
    if (name.startsWith("past_conv")) {
      cache[name] = new ort.Tensor(
        "float32",
        new Float32Array(HIDDEN_SIZE * CONV_CACHE_LEN),
        [1, HIDDEN_SIZE, CONV_CACHE_LEN]
      );
    } else if (name.startsWith("past_key_values")) {
      cache[name] = new ort.Tensor(
        "float32",
        new Float32Array(0),
        [1, NUM_KV_HEADS, 0, HEAD_DIM]
      );
    }
  }
  return cache;
}

// Update cache: present_* → past_*
function updateCache(cache, outputs) {
  for (const [name, tensor] of Object.entries(outputs)) {
    if (name.startsWith("present_conv")) {
      cache[name.replace("present_conv", "past_conv")] = tensor;
    } else if (name.startsWith("present.")) {
      cache[name.replace("present.", "past_key_values.")] = tensor;
    }
  }
}

// Greedy argmax of last token logits
function argmaxLast(logits) {
  const vocabSize = logits.dims[2];
  const offset = (logits.dims[1] - 1) * vocabSize;
  const data = logits.data;
  let maxIdx = 0;
  let maxVal = data[offset];
  for (let i = 1; i < vocabSize; i++) {
    if (data[offset + i] > maxVal) {
      maxVal = data[offset + i];
      maxIdx = i;
    }
  }
  return maxIdx;
}

self.onmessage = async (e) => {
  const { type, data } = e.data;

  if (type === "check") {
    const cached = await checkModelCached();
    self.postMessage({ type: "cache_status", data: { cached } });
    return;
  }

  if (type === "load") {
    try {
      self.postMessage({ type: "status", data: "Loading tokenizer..." });
      tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);

      self.postMessage({ type: "status", data: "Loading text embeddings (fp16)..." });
      embedTokens = await loadSession("embed_tokens_fp16", 1, "Embed tokens fp16");

      self.postMessage({ type: "status", data: "Loading decoder (q4, ~1.2GB)..." });
      decoder = await loadSession("decoder_q4", 1, "Decoder q4");

      console.log("[worker-vl] decoder inputNames:", decoder.inputNames);
      console.log("[worker-vl] decoder outputNames:", decoder.outputNames);

      self.postMessage({ type: "loaded" });
    } catch (err) {
      console.error("[worker-vl] load error:", err);
      self.postMessage({ type: "error", data: err.message });
    }
    return;
  }

  if (type === "generate") {
    if (!tokenizer || !decoder || !embedTokens) {
      self.postMessage({ type: "error", data: "Model not loaded" });
      return;
    }

    try {
      const messages = data.messages || [];

      // Build prompt using chat template
      const prompt = tokenizer.apply_chat_template(messages, {
        add_generation_prompt: true,
        tokenize: false,
      });
      const inputIds = tokenizer.encode(prompt);

      console.log("[worker-vl] prompt length:", inputIds.length, "tokens");
      self.postMessage({
        type: "generate_start",
        data: { promptTokens: inputIds.length },
      });

      // Get initial embeddings for all input tokens
      let embeds = await getTextEmbeddings(inputIds);
      let curLen = inputIds.length;

      const cache = initCache();
      const maxTokens = data.maxTokens || 2048;
      const eosTokenId = tokenizer.eos_token_id ?? 7;
      const startTime = performance.now();
      let totalTokens = 0;

      for (let step = 0; step < maxTokens; step++) {
        const attentionMask = new ort.Tensor(
          "int64",
          new BigInt64Array(curLen).fill(1n),
          [1, curLen]
        );

        const outputs = await decoder.run({
          inputs_embeds: embeds,
          attention_mask: attentionMask,
          ...cache,
        });

        const nextToken = argmaxLast(outputs.logits);
        totalTokens++;

        // Stream decoded token
        const text = tokenizer.decode([nextToken], { skip_special_tokens: false });
        const elapsed = (performance.now() - startTime) / 1000;
        self.postMessage({
          type: "token",
          data: {
            text,
            tokenCount: totalTokens,
            tokensPerSec: totalTokens / elapsed,
          },
        });

        if (nextToken === eosTokenId) break;

        updateCache(cache, outputs);
        embeds = await getTextEmbeddings([nextToken]);
        curLen++;
      }

      const elapsed = (performance.now() - startTime) / 1000;
      self.postMessage({
        type: "generate_done",
        data: {
          tokenCount: totalTokens,
          elapsed,
          tokensPerSec: totalTokens / elapsed,
        },
      });
    } catch (err) {
      console.error("[worker-vl] generate error:", err);
      self.postMessage({ type: "error", data: err.message });
    }
  }
};

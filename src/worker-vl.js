import * as ort from "onnxruntime-web/webgpu";
import { AutoTokenizer } from "@huggingface/transformers";

const MODEL_ID = "LiquidAI/LFM2.5-VL-1.6B-ONNX";
const MODEL_BASE = `https://huggingface.co/${MODEL_ID}/resolve/main/onnx`;

// Architecture constants (from config.json → text_config)
const HIDDEN_SIZE = 2048;
const NUM_KV_HEADS = 8;
const HEAD_DIM = 64; // 2048 / 32
const CONV_CACHE_LEN = 3;

// Vision constants (from processor_config.json)
const PATCH_SIZE = 16;
const DOWNSAMPLE_FACTOR = 2;
const GRID_UNIT = PATCH_SIZE * DOWNSAMPLE_FACTOR; // 32
const MAX_IMAGE_TOKENS = 256;
const MIN_IMAGE_TOKENS = 64;
const MAX_PATCHES_PER_TILE = 1024; // (512/16)^2
const IMAGE_TOKEN_ID = 396; // <image> token

let tokenizer = null;
let embedTokens = null;
let embedImages = null;
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
  if (cached) return cached.arrayBuffer();

  // Fetch with progress
  console.log("[worker-vl] downloading:", label || stableUrl);
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

// Smart resize: find dimensions that are multiples of 32, within token budget, preserving aspect ratio
function smartResize(w, h) {
  let cols = Math.ceil(w / GRID_UNIT);
  let rows = Math.ceil(h / GRID_UNIT);
  let tokens = cols * rows;

  if (tokens > MAX_IMAGE_TOKENS) {
    const scale = Math.sqrt(MAX_IMAGE_TOKENS / tokens);
    cols = Math.max(1, Math.round(cols * scale));
    rows = Math.max(1, Math.round(rows * scale));
  } else if (tokens < MIN_IMAGE_TOKENS) {
    const scale = Math.sqrt(MIN_IMAGE_TOKENS / tokens);
    cols = Math.max(1, Math.round(cols * scale));
    rows = Math.max(1, Math.round(rows * scale));
  }

  return [cols * GRID_UNIT, rows * GRID_UNIT];
}

// Preprocess image dataUrl → tensors for embed_images
// Returns { pixelValues, pixelAttentionMask, spatialShapes, patchRows, patchCols }
async function preprocessImage(dataUrl) {
  // Decode image in worker via fetch + createImageBitmap
  const response = await fetch(dataUrl);
  const blob = await response.blob();

  // Get original dimensions
  const origBitmap = await createImageBitmap(blob);
  const origW = origBitmap.width, origH = origBitmap.height;
  const [newW, newH] = smartResize(origW, origH);
  origBitmap.close();

  console.log("[worker-vl] image resize: %dx%d → %dx%d", origW, origH, newW, newH);

  // Resize and draw to OffscreenCanvas
  const bitmap = await createImageBitmap(blob, { resizeWidth: newW, resizeHeight: newH, resizeQuality: "high" });
  const canvas = new OffscreenCanvas(newW, newH);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();

  const imageData = ctx.getImageData(0, 0, newW, newH);
  const pixels = imageData.data; // Uint8ClampedArray RGBA

  // Extract 16×16 patches and normalize
  const patchRows = newH / PATCH_SIZE;
  const patchCols = newW / PATCH_SIZE;
  const totalPatches = patchRows * patchCols;

  // Padded to MAX_PATCHES_PER_TILE
  const pixelValues = new Float32Array(MAX_PATCHES_PER_TILE * 768);

  for (let pr = 0; pr < patchRows; pr++) {
    for (let pc = 0; pc < patchCols; pc++) {
      const patchIdx = pr * patchCols + pc;
      const patchOffset = patchIdx * 768;
      let k = 0;

      // Extract 16×16 patch, RGB only, normalized to [-1, 1]
      for (let py = 0; py < PATCH_SIZE; py++) {
        for (let px = 0; px < PATCH_SIZE; px++) {
          const imgX = pc * PATCH_SIZE + px;
          const imgY = pr * PATCH_SIZE + py;
          const rgbaIdx = (imgY * newW + imgX) * 4;
          pixelValues[patchOffset + k++] = pixels[rgbaIdx] / 127.5 - 1.0;     // R
          pixelValues[patchOffset + k++] = pixels[rgbaIdx + 1] / 127.5 - 1.0; // G
          pixelValues[patchOffset + k++] = pixels[rgbaIdx + 2] / 127.5 - 1.0; // B
        }
      }
    }
  }

  // Attention mask: 1 for valid patches, 0 for padding
  const attentionMask = new BigInt64Array(MAX_PATCHES_PER_TILE);
  for (let i = 0; i < totalPatches; i++) attentionMask[i] = 1n;

  // Spatial shapes: [patchRows, patchCols]
  const spatialShapes = new BigInt64Array([BigInt(patchRows), BigInt(patchCols)]);

  return {
    pixelValues: new ort.Tensor("float32", pixelValues, [1, MAX_PATCHES_PER_TILE, 768]),
    pixelAttentionMask: new ort.Tensor("int64", attentionMask, [1, MAX_PATCHES_PER_TILE]),
    spatialShapes: new ort.Tensor("int64", spatialShapes, [1, 2]),
    totalPatches,
  };
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

      self.postMessage({ type: "status", data: "Loading image encoder (q4)..." });
      embedImages = await loadSession("embed_images_q4", 1, "Image encoder q4");

      self.postMessage({ type: "status", data: "Loading decoder (q4, ~1.2GB)..." });
      decoder = await loadSession("decoder_q4", 1, "Decoder q4");

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

      // Check if any message has an image
      const imageMsg = messages.find((m) => m.image);
      let imageEmbeds = null;
      let numImageTokens = 0;

      if (imageMsg && embedImages) {
        self.postMessage({ type: "status", data: "Processing image..." });
        const imgTensors = await preprocessImage(imageMsg.image);

        const imgOut = await embedImages.run({
          pixel_values: imgTensors.pixelValues,
          pixel_attention_mask: imgTensors.pixelAttentionMask,
          spatial_shapes: imgTensors.spatialShapes,
        });

        const outName = Object.keys(imgOut)[0];
        imageEmbeds = imgOut[outName];
        numImageTokens = imageEmbeds.dims[0];
        console.log("[worker-vl] image → %d tokens", numImageTokens);
      }

      // Build prompt manually — tokenizer has no chat_template set
      let prompt = "<|startoftext|>";
      prompt += "<|im_start|>system\nYou are a helpful multimodal assistant by Liquid AI.<|im_end|>\n";
      for (const msg of messages) {
        if (msg.role === "system") continue;
        // Insert <image> tokens before text for the message that has an image
        const imagePrefix = (msg.image && numImageTokens > 0) ? "<image>".repeat(numImageTokens) : "";
        prompt += `<|im_start|>${msg.role}\n${imagePrefix}${msg.content}<|im_end|>\n`;
      }
      prompt += "<|im_start|>assistant\n";
      const inputIds = tokenizer.encode(prompt);

      console.log("[worker-vl] prompt: %d tokens (%d image)", inputIds.length, numImageTokens);
      self.postMessage({
        type: "generate_start",
        data: { promptTokens: inputIds.length },
      });

      // Get text embeddings, then merge image embeddings at <image> positions
      let embeds = await getTextEmbeddings(inputIds);

      if (imageEmbeds && numImageTokens > 0) {
        // Replace <image> token positions with image embeddings
        const embedsData = embeds.data; // Float32Array [1, seqLen, 2048]
        const imgData = imageEmbeds.data; // Float32Array [numImageTokens, 2048]
        let imgIdx = 0;
        for (let i = 0; i < inputIds.length && imgIdx < numImageTokens; i++) {
          if (inputIds[i] === IMAGE_TOKEN_ID) {
            const dst = i * HIDDEN_SIZE;
            const src = imgIdx * HIDDEN_SIZE;
            for (let j = 0; j < HIDDEN_SIZE; j++) {
              embedsData[dst + j] = imgData[src + j];
            }
            imgIdx++;
          }
        }
        console.log("[worker-vl] merged %d image embeddings", imgIdx);
      }

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

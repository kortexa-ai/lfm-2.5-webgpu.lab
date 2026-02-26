import * as ort from "onnxruntime-web/webgpu";
import { AutoTokenizer } from "@huggingface/transformers";

const MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B-ONNX";
const MODEL_BASE = `https://huggingface.co/${MODEL_ID}/resolve/main/onnx`;

// Architecture constants (from config.json → lfm)
const HIDDEN_SIZE = 2048;
const NUM_KV_HEADS = 8;
const HEAD_DIM = 64;
const CONV_CACHE_LEN = 3;

// Audio constants (from audio_embedding.json + config.json)
const NUM_CODEBOOKS = 8;
const CODEBOOK_VOCAB = 2049; // 0-2047 audio tokens, 2048 = end-of-audio
const AUDIO_END_TOKEN = 2048;
const AUDIO_START_TOKEN_ID = 128; // <|audio_start|>

// Audio output config
const OUTPUT_SAMPLE_RATE = 24000;

let tokenizer = null;
let embedWeight = null; // Float32Array from embed_tokens.bin
let embedHiddenSize = 0; // from embed_tokens.json
let decoder = null;
let audioEmbedding = null;
let detokenizer = null;
let depthformer = null;

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
    console.log("[worker-audio] cache hit:", stableUrl);
    return cached.arrayBuffer();
  }

  // Fetch with progress
  console.log("[worker-audio] downloading:", stableUrl);
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

// Get text embeddings by looking up from the binary weight table
function getTextEmbeddings(ids) {
  const hs = embedHiddenSize;
  const embeds = new Float32Array(ids.length * hs);
  for (let i = 0; i < ids.length; i++) {
    const offset = ids[i] * hs;
    embeds.set(embedWeight.subarray(offset, offset + hs), i * hs);
  }
  return new ort.Tensor("float32", embeds, [1, ids.length, hs]);
}

// Initialize KV cache from decoder input names
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

// Top-K sampling from logits array
function sampleTopK(logitsData, vocabSize, offset, temperature = 0.8, topK = 64) {
  // Collect top-K indices
  const indices = [];
  for (let i = 0; i < vocabSize; i++) {
    indices.push({ idx: i, val: logitsData[offset + i] });
  }
  indices.sort((a, b) => b.val - a.val);
  const topIndices = indices.slice(0, topK);

  // Apply temperature and softmax
  const maxVal = topIndices[0].val;
  const exps = topIndices.map((x) => Math.exp((x.val - maxVal) / temperature));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((e) => e / sumExp);

  // Multinomial sample
  const r = Math.random();
  let cum = 0;
  for (let i = 0; i < probs.length; i++) {
    cum += probs[i];
    if (r < cum) return topIndices[i].idx;
  }
  return topIndices[topIndices.length - 1].idx;
}

// Depthformer config (from config.json → depthformer)
const DP_DIM = 1024;
const DP_LAYERS = 6;
const DP_NUM_KV_HEADS = 8;
const DP_HEAD_DIM = 32; // DP_DIM / 32_query_heads = 32

// Generate one audio frame: 8 codebook values via depthformer
// Shapes from Liquid4All/onnx-export depthformer_builder.py:
//   hidden_states:    [1, 2048] float32
//   depth_slices_in:  [1, 8, 1024] float32  (batch, codebooks, dim)
//   step_idx:         [] scalar int64
//   prev_token:       [1] int64
//   past_keys:        [6, 1, 8, past_len, 32] float32  (layers, batch, kv_heads, seq, head_dim)
//   past_values:      [6, 1, 8, past_len, 32] float32
//   seqlens_k:        [1] int32
//   total_seq_len:    [] scalar int32
async function generateAudioFrame(hiddenStates) {
  const frameCodes = new Array(NUM_CODEBOOKS);

  // Initialize depthformer state for this frame (fresh cache per frame)
  // Pre-allocate KV cache to max_seq_len (past_present_share_buffer pattern)
  const MAX_DP_SEQ = 16;
  let depthSlices = new ort.Tensor("float32", new Float32Array(1 * NUM_CODEBOOKS * DP_DIM), [1, NUM_CODEBOOKS, DP_DIM]);
  let pastKeys = new ort.Tensor("float32", new Float32Array(DP_LAYERS * 1 * DP_NUM_KV_HEADS * MAX_DP_SEQ * DP_HEAD_DIM), [DP_LAYERS, 1, DP_NUM_KV_HEADS, MAX_DP_SEQ, DP_HEAD_DIM]);
  let pastValues = new ort.Tensor("float32", new Float32Array(DP_LAYERS * 1 * DP_NUM_KV_HEADS * MAX_DP_SEQ * DP_HEAD_DIM), [DP_LAYERS, 1, DP_NUM_KV_HEADS, MAX_DP_SEQ, DP_HEAD_DIM]);
  let seqlensK = new ort.Tensor("int32", new Int32Array([0]), [1]);
  let totalSeqLen = new ort.Tensor("int32", new Int32Array([1]), []); // scalar, past+1
  let prevToken = new ort.Tensor("int64", new BigInt64Array([0n]), [1]);

  for (let cb = 0; cb < NUM_CODEBOOKS; cb++) {
    const feeds = {
      hidden_states: new ort.Tensor("float32", new Float32Array(hiddenStates), [1, HIDDEN_SIZE]),
      depth_slices_in: depthSlices,
      step_idx: new ort.Tensor("int64", new BigInt64Array([BigInt(cb)]), []), // scalar
      prev_token: prevToken,
      past_keys: pastKeys,
      past_values: pastValues,
      seqlens_k: seqlensK,
      total_seq_len: totalSeqLen,
    };

    const outputs = await depthformer.run(feeds);

    // Sample from depthformer logits
    const logits = outputs.logits;
    const vocabSize = logits.dims[logits.dims.length - 1];
    frameCodes[cb] = sampleTopK(logits.data, vocabSize, 0, 0.8, 64);

    // Update depthformer state for next codebook step
    depthSlices = outputs.depth_slices;
    pastKeys = outputs.new_keys;
    pastValues = outputs.new_values;
    prevToken = new ort.Tensor("int64", new BigInt64Array([BigInt(frameCodes[cb])]), [1]);
    seqlensK = new ort.Tensor("int32", new Int32Array([cb + 1]), [1]);
    totalSeqLen = new ort.Tensor("int32", new Int32Array([cb + 2]), []); // scalar, past+1
  }

  return frameCodes;
}

// Sum audio embeddings across codebooks → single embedding
function sumEmbeddings(embedTensor) {
  const data = embedTensor.data;
  const total = data.length;
  const summed = new Float32Array(HIDDEN_SIZE);
  for (let i = 0; i < total; i++) {
    summed[i % HIDDEN_SIZE] += data[i];
  }
  return new ort.Tensor("float32", summed, [1, 1, HIDDEN_SIZE]);
}

// ISTFT constants (from reference: Liquid4All/onnx-export infer.py)
const ISTFT_N_FFT = 1280;
const ISTFT_HOP = 320;
const ISTFT_WIN = 1280;
const ISTFT_N_FREQ = ISTFT_N_FFT / 2 + 1; // 641
const ISTFT_PAD = (ISTFT_WIN - ISTFT_HOP) / 2; // 480 — "same" padding trim

// Inverse Real FFT for a single frame: freq-domain (641 complex bins) → time-domain (1280 samples)
// Matches np.fft.irfft(spec, n_fft, norm="backward")
// Uses Hermitian symmetry: bins N/2+1..N-1 are conjugate of bins N/2-1..1
function irfft(real, imag, nFft) {
  const nFreq = nFft / 2 + 1; // 641
  const out = new Float64Array(nFft); // float64 for precision
  for (let n = 0; n < nFft; n++) {
    // DC bin (k=0): no conjugate pair
    let sum = real[0];
    // Nyquist bin (k=N/2): no conjugate pair
    sum += real[nFreq - 1] * (n % 2 === 0 ? 1 : -1); // cos(πn) = (-1)^n
    // Bins 1..N/2-1: each has a conjugate pair at N-k, contributing 2*Re(X[k]*e^{i2πkn/N})
    for (let k = 1; k < nFreq - 1; k++) {
      const angle = (2 * Math.PI * k * n) / nFft;
      sum += 2 * (real[k] * Math.cos(angle) - imag[k] * Math.sin(angle));
    }
    out[n] = sum / nFft;
  }
  return out;
}

// Decode audio codes → waveform via detokenizer + ISTFT
async function decodeAudioCodes(audioCodes) {
  const numFrames = audioCodes.length;
  const codesArray = new BigInt64Array(numFrames * NUM_CODEBOOKS);

  // Layout must be codebook-major to match tensor shape [1, NUM_CODEBOOKS, numFrames]
  for (let cb = 0; cb < NUM_CODEBOOKS; cb++) {
    for (let i = 0; i < numFrames; i++) {
      codesArray[cb * numFrames + i] = BigInt(audioCodes[i][cb]);
    }
  }

  console.log("[worker-audio] decoding", numFrames, "audio frames");

  const detokOut = await detokenizer.run({
    audio_codes: new ort.Tensor("int64", codesArray, [1, NUM_CODEBOOKS, numFrames]),
  });

  const outputNames = Object.keys(detokOut);
  console.log("[worker-audio] detokenizer outputs:", outputNames, outputNames.map((n) => detokOut[n].dims));

  const stftFeatures = detokOut.stft_features || detokOut[outputNames[0]];
  const dims = stftFeatures.dims;
  const stftData = stftFeatures.data;

  // stft_features shape: [1, T_stft, 1282] where 1282 = 2 * 641 (log_magnitude + phase)
  const tStft = dims[1];
  const featureDim = dims[2];
  console.log("[worker-audio] STFT: %d frames, %d features (expected %d)", tStft, featureDim, ISTFT_N_FREQ * 2);

  // Log first frame's STFT values for diagnostics
  if (tStft > 0) {
    const logMag0 = stftData[0], logMag1 = stftData[1], logMag640 = stftData[640];
    const angle0 = stftData[ISTFT_N_FREQ], angle1 = stftData[ISTFT_N_FREQ + 1];
    console.log("[worker-audio] STFT frame 0 sample: logMag[0..2,640]=[%.4f,%.4f,%.4f] angle[0..1]=[%.4f,%.4f]",
      logMag0, logMag1, logMag640, angle0, angle1);
  }

  // Reconstruct waveform via overlap-add ISTFT
  // Matches reference: Liquid4All/onnx-export infer.py _istft_same_padding()
  const window = hanningWindow(ISTFT_WIN);
  const outputLen = (tStft - 1) * ISTFT_HOP + ISTFT_WIN;
  const waveform = new Float64Array(outputLen); // float64 for precision
  const windowEnvelope = new Float64Array(outputLen);

  // Pre-compute window squared
  const windowSq = new Float64Array(ISTFT_WIN);
  for (let i = 0; i < ISTFT_WIN; i++) windowSq[i] = window[i] * window[i];

  for (let t = 0; t < tStft; t++) {
    const rowOffset = t * featureDim;

    // Split into log_magnitude and phase angle
    // Reconstruct complex STFT: magnitude * exp(i * angle)
    const real = new Float64Array(ISTFT_N_FREQ);
    const imag = new Float64Array(ISTFT_N_FREQ);
    for (let k = 0; k < ISTFT_N_FREQ; k++) {
      const logMag = stftData[rowOffset + k];
      const angle = stftData[rowOffset + ISTFT_N_FREQ + k];
      const mag = Math.exp(logMag);
      real[k] = mag * Math.cos(angle);
      imag[k] = mag * Math.sin(angle);
    }

    // IRFFT → time-domain frame, then apply window
    const frame = irfft(real, imag, ISTFT_N_FFT);

    // Overlap-add: windowed IRFFT result + window² envelope
    const start = t * ISTFT_HOP;
    for (let n = 0; n < ISTFT_WIN; n++) {
      waveform[start + n] += frame[n] * window[n];
      windowEnvelope[start + n] += windowSq[n];
    }
  }

  // Trim "same" padding, then normalize by window envelope
  const trimmed = new Float32Array(outputLen - 2 * ISTFT_PAD);
  for (let i = 0; i < trimmed.length; i++) {
    const j = i + ISTFT_PAD;
    trimmed[i] = windowEnvelope[j] > 1e-8 ? waveform[j] / windowEnvelope[j] : 0;
  }

  // Normalize amplitude to 0.9 peak
  let maxAbs = 0;
  for (let i = 0; i < trimmed.length; i++) {
    const abs = Math.abs(trimmed[i]);
    if (abs > maxAbs) maxAbs = abs;
  }
  if (maxAbs > 0) {
    const scale = 0.9 / maxAbs;
    for (let i = 0; i < trimmed.length; i++) {
      trimmed[i] *= scale;
    }
  }

  console.log("[worker-audio] ISTFT: %d stft frames → %d samples (%.2fs at %dHz)",
    tStft, trimmed.length, trimmed.length / OUTPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE);

  return trimmed;
}

// Hanning window — matches np.hanning(N): 0.5 - 0.5*cos(2πn/(N-1))
function hanningWindow(size) {
  const w = new Float64Array(size);
  for (let i = 0; i < size; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (size - 1)));
  }
  return w;
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

      self.postMessage({ type: "status", data: "Loading text embeddings (binary)..." });
      const [embedBuf, embedMetaBuf] = await Promise.all([
        cachedFetch(`${MODEL_BASE}/embed_tokens.bin`, "embed_tokens.bin (~128MB)"),
        cachedFetch(`${MODEL_BASE}/embed_tokens.json`, "embed_tokens.json"),
      ]);
      const embedMeta = JSON.parse(new TextDecoder().decode(embedMetaBuf));
      embedWeight = new Float32Array(embedBuf);
      embedHiddenSize = embedMeta.hidden_size;
      console.log("[worker-audio] embed_tokens: vocab_size=%d, hidden_size=%d", embedMeta.vocab_size, embedHiddenSize);

      self.postMessage({ type: "status", data: "Loading decoder (q4, ~1.5GB)..." });
      decoder = await loadSession("decoder_q4", 1, "Decoder q4");
      console.log("[worker-audio] decoder inputNames:", decoder.inputNames);
      console.log("[worker-audio] decoder outputNames:", decoder.outputNames);

      self.postMessage({ type: "status", data: "Loading audio embedding (q4)..." });
      audioEmbedding = await loadSession("audio_embedding_q4", 1, "Audio embedding q4");
      console.log("[worker-audio] audioEmbedding inputNames:", audioEmbedding.inputNames);

      self.postMessage({ type: "status", data: "Loading detokenizer (q4)..." });
      detokenizer = await loadSession("audio_detokenizer_q4", 1, "Detokenizer q4");

      self.postMessage({ type: "status", data: "Loading depthformer (q4)..." });
      depthformer = await loadSession("vocoder_depthformer_q4", 1, "Depthformer q4");
      console.log("[worker-audio] depthformer inputNames:", depthformer.inputNames);
      console.log("[worker-audio] depthformer outputNames:", depthformer.outputNames);

      self.postMessage({ type: "loaded" });
    } catch (err) {
      console.error("[worker-audio] load error:", err);
      self.postMessage({ type: "error", data: err.message });
    }
    return;
  }

  if (type === "generate") {
    if (!tokenizer || !decoder) {
      self.postMessage({ type: "error", data: "Model not loaded" });
      return;
    }

    try {
      const messages = data.messages || [];

      // Audio tokenizer has no chat_template — build prompt manually
      // TTS system prompt (from HF example) triggers audio generation mode
      const TTS_SYSTEM = "Perform TTS. Use the UK female voice.";
      let prompt = "<|startoftext|>";
      prompt += `<|im_start|>system\n${TTS_SYSTEM}<|im_end|>\n`;
      for (const msg of messages) {
        if (msg.role === "system") continue; // skip any user-provided system msgs
        prompt += `<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`;
      }
      prompt += "<|im_start|>assistant\n";
      const inputIds = tokenizer.encode(prompt);

      console.log("[worker-audio] prompt:", JSON.stringify(prompt));
      console.log("[worker-audio] inputIds:", inputIds.length, "tokens");
      self.postMessage({
        type: "generate_start",
        data: { promptTokens: inputIds.length },
      });

      // Get initial embeddings
      let embeds = getTextEmbeddings(inputIds);
      console.log("[worker-audio] embeds shape:", embeds.dims, "dtype:", embeds.type);
      let curLen = inputIds.length;

      const cache = initCache();
      const maxTokens = data.maxTokens || 2048;
      const eosTokenId = tokenizer.eos_token_id ?? 7;
      const startTime = performance.now();
      let totalTokens = 0;

      // Audio state
      let inAudioMode = false;
      const audioCodes = [];

      for (let step = 0; step < maxTokens; step++) {
        const attentionMask = new ort.Tensor(
          "int64",
          new BigInt64Array(curLen).fill(1n),
          [1, curLen]
        );

        console.log("[worker-audio] step %d: running decoder (curLen=%d, embeds=%s)...", step, curLen, embeds.dims);
        const t0 = performance.now();
        const outputs = await decoder.run({
          inputs_embeds: embeds,
          attention_mask: attentionMask,
          ...cache,
        });
        console.log("[worker-audio] step %d: decoder done in %.1fs, outputs: %s", step, (performance.now() - t0) / 1000, Object.keys(outputs));

        updateCache(cache, outputs);

        if (inAudioMode) {
          // Audio generation: use hidden_states from decoder
          const hiddenStates = outputs.hidden_states;
          if (!hiddenStates) {
            console.error("[worker-audio] decoder did not output hidden_states! outputs:", Object.keys(outputs));
            break;
          }

          // Extract last hidden state
          const lastHidden = hiddenStates.data.slice(-HIDDEN_SIZE);

          // Generate one audio frame (8 codebook values)
          const frameCodes = await generateAudioFrame(lastHidden);
          console.log("[worker-audio] audio frame", audioCodes.length, "codes:", frameCodes);

          // Check for end-of-audio
          if (frameCodes[0] === AUDIO_END_TOKEN) {
            console.log("[worker-audio] end-of-audio after", audioCodes.length, "frames");
            break;
          }

          audioCodes.push(frameCodes);
          totalTokens++;

          const elapsed = (performance.now() - startTime) / 1000;
          self.postMessage({
            type: "token",
            data: {
              text: "", // No text during audio gen
              tokenCount: totalTokens,
              tokensPerSec: totalTokens / elapsed,
              audioFrame: audioCodes.length,
            },
          });

          // Get audio embedding for next decoder step
          // Each codebook value maps to: cb_index * codebook_vocab + code
          const audioTokenIds = frameCodes.map((code, cb) => cb * CODEBOOK_VOCAB + code);
          const audioEmbedsResult = await audioEmbedding.run({
            audio_codes: new ort.Tensor(
              "int64",
              new BigInt64Array(audioTokenIds.map(BigInt)),
              [1, NUM_CODEBOOKS]
            ),
          });

          // Sum embeddings across codebooks
          const outputName = Object.keys(audioEmbedsResult)[0];
          embeds = sumEmbeddings(audioEmbedsResult[outputName]);
        } else {
          // Text generation mode
          const nextToken = argmaxLast(outputs.logits);
          totalTokens++;
          const decoded = tokenizer.decode([nextToken], { skip_special_tokens: false });
          console.log("[worker-audio] step %d: token=%d decoded=%s", step, nextToken, JSON.stringify(decoded));

          if (nextToken === AUDIO_START_TOKEN_ID) {
            console.log("[worker-audio] <|audio_start|> detected, switching to audio mode");
            inAudioMode = true;
            self.postMessage({
              type: "token",
              data: {
                text: "",
                tokenCount: totalTokens,
                tokensPerSec: totalTokens / (performance.now() - startTime) * 1000,
                audioStarted: true,
              },
            });
            // Feed the audio_start token embedding back
            embeds = getTextEmbeddings([nextToken]);
          } else {
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
            embeds = getTextEmbeddings([nextToken]);
          }
        }

        curLen++;
      }

      // If we collected audio codes, decode them to waveform
      if (audioCodes.length > 0) {
        self.postMessage({ type: "status", data: "Decoding audio..." });

        try {
          const waveform = await decodeAudioCodes(audioCodes);
          console.log("[worker-audio] waveform:", waveform.length, "samples at", OUTPUT_SAMPLE_RATE, "Hz");
          self.postMessage({
            type: "audio_data",
            data: {
              waveform: waveform,
              sampleRate: OUTPUT_SAMPLE_RATE,
              frames: audioCodes.length,
            },
          });
        } catch (err) {
          console.error("[worker-audio] audio decode error:", err);
          self.postMessage({
            type: "error",
            data: `Audio decode failed: ${err.message}`,
          });
        }
      }

      const elapsed = (performance.now() - startTime) / 1000;
      self.postMessage({
        type: "generate_done",
        data: {
          tokenCount: totalTokens,
          elapsed,
          tokensPerSec: totalTokens / elapsed,
          audioFrames: audioCodes.length,
        },
      });
    } catch (err) {
      console.error("[worker-audio] generate error:", err);
      self.postMessage({ type: "error", data: err.message });
    }
  }
};

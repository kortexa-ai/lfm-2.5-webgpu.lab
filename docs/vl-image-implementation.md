# VL Model — Image Input Implementation

## Architecture Overview

The VL model uses three ONNX sessions:
- **embed_tokens_fp16** — token ID → text embeddings [1, seq_len, 2048]
- **embed_images_q4** (or fp16) — pixel patches → image embeddings [num_image_tokens, 2048]
- **decoder_q4** — autoregressive generation from combined embeddings

Image and text embeddings live in the same 2048-dim space (the vision encoder includes a projector from SigLIP2's 1152-dim to the decoder's 2048-dim).

## How It Works

1. Tokenize the prompt with `<image>` placeholder tokens (token ID **396**)
2. Run `embed_tokens` on all token IDs → text embeddings
3. Preprocess image → pixel patches tensor
4. Run `embed_images` on pixel patches → image embeddings
5. Replace each `<image>` position in text embeddings with corresponding image embedding
6. Feed combined embeddings to decoder as `inputs_embeds`

## Chat Template

```
<|startoftext|><|im_start|>system
You are a helpful multimodal assistant by Liquid AI.<|im_end|>
<|im_start|>user
<image><image>...<image>What is in this image?<|im_end|>
<|im_start|>assistant
```

The number of `<image>` tokens must exactly match the number of embedding vectors output by `embed_images`.

## Image Preprocessing Pipeline

### Config Constants
```
tile_size = 512          // pixels per tile edge
patch_size = 16          // pixels per patch edge
downsample_factor = 2    // 2×2 patch blocks merged into 1 token
max_image_tokens = 256   // per tile
min_image_tokens = 64
max_tiles = 10
min_tiles = 2
use_thumbnail = true
max_patches_per_tile = 1024  // (512/16)^2
```

### Step 1: Smart Resize (single-tile path)

For a demo, start with single-tile images:

1. Ensure dimensions are multiples of 32 (`patch_size × downsample_factor = 16 × 2`)
2. Token count = `(H/32) × (W/32)`, constrained to [64, 256]
3. Preserve aspect ratio; round to nearest 32-pixel boundary

Examples:
- 512×512 → 512×512 → 256 tokens
- 800×600 → 512×384 → 16×12 = 192 tokens
- 1920×1080 → 512×288 → 16×9 = 144 tokens

Algorithm:
```js
function smartResize(w, h) {
  const minTokens = 64, maxTokens = 256, gridUnit = 32;
  // Scale to fit within token budget
  let tokens = Math.ceil(w / gridUnit) * Math.ceil(h / gridUnit);
  let scale = 1;
  if (tokens > maxTokens) scale = Math.sqrt(maxTokens / tokens);
  if (tokens < minTokens) scale = Math.sqrt(minTokens / tokens);
  let nw = Math.round(w * scale / gridUnit) * gridUnit;
  let nh = Math.round(h * scale / gridUnit) * gridUnit;
  // Clamp
  nw = Math.max(gridUnit, nw);
  nh = Math.max(gridUnit, nh);
  return [nw, nh];
}
```

### Step 2: Patch Extraction

Divide the resized image into 16×16 pixel patches:
- Rows of patches: `H / 16`
- Cols of patches: `W / 16`
- Total patches: rows × cols (≤ 1024 for 512×512)
- Each patch: flatten to 768 values (16 × 16 × 3 RGB channels)

### Step 3: Normalization

```js
normalized = pixel / 127.5 - 1.0
// Equivalent to: (pixel/255 - 0.5) / 0.5
// Range: [-1.0, 1.0]
```

### Step 4: Build Tensors

```
pixel_values:          Float32 [num_tiles, max_patches_per_tile, 768]  // zero-padded
pixel_attention_mask:  Int64   [num_tiles, max_patches_per_tile]       // 1=valid, 0=pad
spatial_shapes:        Int64   [num_tiles, 2]                          // [rows, cols]
```

For single-tile (simplest case):
- num_tiles = 1
- patches = rows × cols
- Pad to 1024 patches with zeros
- attention_mask: 1 for first `patches` entries, 0 for rest
- spatial_shapes: [[rows, cols]]

### Step 5: Image Token Count

After `embed_images` runs, count the output embeddings:
```
num_image_tokens = embed_images_output.dims[0]  // first dim
```

This is the number of `<image>` tokens to insert in the prompt.

## embed_images ONNX Session

**Inputs:**
| Name | Type | Shape |
|------|------|-------|
| pixel_values | float32 | [num_tiles, 1024, 768] |
| pixel_attention_mask | int64 | [num_tiles, 1024] |
| spatial_shapes | int64 | [num_tiles, 2] |

**Output:**
| Name | Type | Shape |
|------|------|-------|
| image_features | float32 | [num_image_tokens, 2048] |

The output is already projected to the decoder's hidden dimension.

## Embedding Merge

```js
const IMAGE_TOKEN_ID = 396;

// 1. Build prompt with N <image> tokens
let userContent = "<image>".repeat(numImageTokens) + userText;

// 2. Tokenize and embed text
const inputIds = tokenizer.encode(prompt);
const textEmbeds = await embedTokens.run({ input_ids: ... });

// 3. Get image embeddings
const imgOut = await embedImages.run({ pixel_values, pixel_attention_mask, spatial_shapes });
const imageEmbeds = imgOut.image_features; // [numImageTokens, 2048]

// 4. Replace <image> positions
const embedsData = textEmbeds.inputs_embeds.data; // Float32Array
let imgIdx = 0;
for (let i = 0; i < inputIds.length; i++) {
  if (inputIds[i] === IMAGE_TOKEN_ID && imgIdx < numImageTokens) {
    const src = imageEmbeds.data.slice(imgIdx * 2048, (imgIdx + 1) * 2048);
    embedsData.set(src, i * 2048);
    imgIdx++;
  }
}
```

## App.jsx Changes

1. Add image upload button (file input or clipboard paste)
2. Read image as ImageBitmap or canvas
3. Send pixel data to worker via postMessage (transferable ArrayBuffer)
4. Worker preprocesses, runs embed_images, merges, generates
5. Show image thumbnail in chat message bubble

## Multi-Tile (Future Enhancement)

For large/high-res images:
1. Find best tile grid (rows × cols) matching aspect ratio
2. Resize to grid_cols×512 × grid_rows×512
3. Split into 512×512 tiles
4. Optionally add thumbnail tile (downscaled full image)
5. Process all tiles as batch: pixel_values shape [num_tiles, 1024, 768]
6. Total image tokens = sum across all tiles

## File Size Estimates

| Session | Size |
|---------|------|
| embed_images_q4 | ~200MB (model + data) |
| embed_images_fp16 | ~800MB (model + data, split files) |

Recommend q4 for WebGPU demo to keep download reasonable.

import {
  AutoTokenizer,
  AutoModelForCausalLM,
  TextStreamer,
} from "@huggingface/transformers";

let tokenizer = null;
let model = null;

const MODEL_ID = "LiquidAI/LFM2.5-1.2B-Thinking-ONNX";

const TOOLS = [
  {
    name: "wave_hello",
    description: "Wave at the user with a friendly greeting. Use this whenever the user says hello or greets you.",
    parameters: {
      type: "object",
      properties: {
        name: {
          type: "string",
          description: "The name of the person to wave at, or 'friend' if unknown",
        },
      },
      required: ["name"],
    },
  },
];

const SYSTEM_PROMPT = `You are a helpful assistant with access to tools.

If the user says hello or greets you, you MUST use the wave_hello tool to wave at them.

List of tools: ${JSON.stringify(TOOLS)}`;

// Parse tool calls from model output
function parseToolCalls(text) {
  const pattern = /\<\|tool_call_start\|\>([\s\S]*?)\<\|tool_call_end\|\>/g;
  const calls = [];
  let match;
  while ((match = pattern.exec(text)) !== null) {
    const raw = match[1].trim();
    // Parse Pythonic call like [wave_hello(name="User")]
    const fnMatch = raw.match(/\[?(\w+)\((.*?)\)\]?/);
    if (fnMatch) {
      const name = fnMatch[1];
      const argsStr = fnMatch[2];
      const args = {};
      // Parse key="value" pairs
      const argPattern = /(\w+)\s*=\s*"([^"]*)"/g;
      let argMatch;
      while ((argMatch = argPattern.exec(argsStr)) !== null) {
        args[argMatch[1]] = argMatch[2];
      }
      calls.push({ name, args, raw });
    }
  }
  return calls;
}

// Execute a tool call
function executeTool(call) {
  if (call.name === "wave_hello") {
    const name = call.args.name || "friend";
    return JSON.stringify({ result: `*waves enthusiastically at ${name}* 👋✨` });
  }
  return JSON.stringify({ error: `Unknown tool: ${call.name}` });
}

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
      // Prepend system message with tool definitions
      const messages = [
        { role: "system", content: SYSTEM_PROMPT },
        ...(data.messages || []),
      ];

      const startTime = performance.now();
      let totalTokens = 0;

      // Generate with possible tool-call loop (max 1 tool round-trip)
      for (let round = 0; round < 2; round++) {
        console.log(`[worker] round ${round}, messages:`, JSON.stringify(messages.map(m => ({ role: m.role, content: m.content.slice(0, 100) }))));

        const inputs = tokenizer.apply_chat_template(messages, {
          add_generation_prompt: true,
          return_dict: true,
        });

        if (round === 0) {
          self.postMessage({ type: "generate_start", data: { promptTokens: inputs.input_ids.dims[1] } });
        }

        let roundText = "";

        const streamer = new TextStreamer(tokenizer, {
          skip_prompt: true,
          skip_special_tokens: false,
          callback_function: (text) => {
            totalTokens++;
            roundText += text;
            const elapsed = (performance.now() - startTime) / 1000;
            self.postMessage({
              type: "token",
              data: {
                text,
                tokenCount: totalTokens,
                tokensPerSec: totalTokens / elapsed,
              },
            });
          },
        });

        await model.generate({
          ...inputs,
          max_new_tokens: data.maxTokens || 2048,
          temperature: data.temperature ?? 0.7,
          do_sample: data.temperature > 0,
          streamer,
        });

        console.log(`[worker] round ${round} raw output:`, JSON.stringify(roundText));

        // Check for tool calls in this round's output
        const toolCalls = parseToolCalls(roundText);
        console.log(`[worker] round ${round} tool calls found:`, toolCalls.length, toolCalls);

        if (toolCalls.length === 0) break;

        // Execute tools and feed results back
        for (const call of toolCalls) {
          const result = executeTool(call);
          console.log(`[worker] executing tool ${call.name}:`, call.args, "=>", result);
          self.postMessage({ type: "tool_call", data: { call, result } });

          messages.push({ role: "assistant", content: roundText });
          messages.push({ role: "tool", content: result });
        }

        // Reset stream for next round — notify UI we're continuing
        self.postMessage({ type: "tool_continue" });
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
      self.postMessage({ type: "error", data: err.message });
    }
  }
};

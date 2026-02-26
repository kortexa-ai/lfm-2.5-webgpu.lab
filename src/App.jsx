import { useState, useRef, useEffect, useCallback } from "react";

const MODELS = [
  {
    id: "LiquidAI/LFM2.5-1.2B-Thinking-ONNX",
    type: "thinking",
    label: "1.2B Thinking",
    desc: "Reasoning model with chain-of-thought",
    size: "~600MB",
  },
  {
    id: "LiquidAI/LFM2.5-VL-1.6B-ONNX",
    type: "vl",
    label: "VL 1.6B",
    desc: "Vision-language text chat",
    size: "~1.5GB",
  },
  {
    id: "LiquidAI/LFM2.5-Audio-1.5B-ONNX",
    type: "audio",
    label: "Audio 1.5B",
    desc: "Text chat + speech output",
    size: "~1.8GB",
  },
];

const STORAGE_KEY = "lfm25-selected-model";

function getSelectedModel() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored && MODELS.find((m) => m.id === stored)) return stored;
  } catch {}
  return MODELS[0].id;
}

function setSelectedModel(id) {
  try { localStorage.setItem(STORAGE_KEY, id); } catch {}
}

function getModelInfo(id) {
  return MODELS.find((m) => m.id === id) || MODELS[0];
}

// Check cache for a single model directly from main thread
async function isModelCached(modelId) {
  const model = MODELS.find((m) => m.id === modelId);
  // ONNX RT models (VL, Audio) — check for decoder file in Cache API
  if (model?.type !== "thinking") {
    try {
      const cache = await caches.open("onnx-model-cache");
      const base = `https://huggingface.co/${modelId}/resolve/main/onnx`;
      const hit = await cache.match(`${base}/decoder_q4.onnx`);
      return !!hit;
    } catch {
      return false;
    }
  }
  // Thinking model uses transformers.js Cache API
  try {
    const cache = await caches.open("transformers-cache");
    const keys = await cache.keys();
    return keys.some((req) => req.url.includes(modelId.replace("/", "%2F")) || req.url.includes(modelId));
  } catch {
    return false;
  }
}

// Check cache for all models
async function checkAllCaches() {
  const result = {};
  for (const m of MODELS) {
    result[m.id] = await isModelCached(m.id);
  }
  return result;
}

function LabHeader() {
  const homeUrl = import.meta.env.DEV
    ? "https://localhost:8030/"
    : "https://lab.kortexa.ai/";

  return (
    <header className="absolute top-0 left-0 right-0 p-6 z-10">
      <a href={homeUrl} className="back-link">
        <img src={`${homeUrl}lab-transparent.png`} alt="" className="logo" />
        <span className="text-sm font-medium uppercase tracking-wider">kortexa.ai lab</span>
        <span className="text-sm">&larr;</span>
      </a>
    </header>
  );
}

function PulseDots() {
  return (
    <span className="inline-flex gap-1 ml-2">
      <span className="pulse-dot" />
      <span className="pulse-dot" />
      <span className="pulse-dot" />
    </span>
  );
}

function ProgressBar({ progress }) {
  if (!progress || !progress.total) return null;
  const pct = Math.round((progress.loaded / progress.total) * 100);
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-neutral-500">
        <span>{progress.file || "Loading..."}</span>
        <span>{pct}%</span>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function parseThinking(content) {
  if (!content.includes("<think>") && "<think>".startsWith(content.trim())) {
    return { thinking: null, reply: "", thinkingDone: false };
  }

  if (!content.includes("<think>")) {
    return { thinking: null, reply: cleanSpecialTokens(content), thinkingDone: true };
  }

  const afterOpen = content.split("<think>")[1];

  if (afterOpen.includes("</think>")) {
    const [thinking, ...rest] = afterOpen.split("</think>");
    return { thinking: cleanSpecialTokens(thinking.trim()), reply: cleanSpecialTokens(rest.join("</think>").trim()), thinkingDone: true };
  }

  return { thinking: cleanSpecialTokens(afterOpen.trim()), reply: "", thinkingDone: false };
}

function cleanSpecialTokens(text) {
  return text
    .replace(/<\|tool_call_start\|>[\s\S]*?<\|tool_call_end\|>/g, "")
    .replace(/<\|tool_call_start\|>[\s\S]*/g, "")
    .replace(/<\|[a-z_]+\|>/g, "")
    .replace(/◁[a-z_]+▷/g, "")
    .trim();
}

function ThinkingBlock({ thinking }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="think-block mb-3">
      <button
        onClick={() => setOpen(!open)}
        className="think-toggle"
      >
        <span className="think-icon">{open ? "▾" : "▸"}</span>
        <span>Thought for a moment</span>
      </button>
      {open && (
        <div className="think-content">
          {thinking}
        </div>
      )}
    </div>
  );
}

function AssistantMessage({ content, isStreaming }) {
  const { thinking, reply, thinkingDone } = parseThinking(content);
  const stillThinking = isStreaming && thinking && !thinkingDone;

  if (stillThinking) {
    return (
      <div className="message message-assistant">
        <span className="text-neutral-400 text-xs font-medium">Thinking</span>
        <PulseDots />
      </div>
    );
  }

  const hasToolMarkers = content.includes("<|tool_call") || content.includes("tool_call");
  const isEmpty = !reply && !isStreaming && !thinking;

  return (
    <div className="message message-assistant">
      {thinking && <ThinkingBlock thinking={thinking} />}
      {reply ? (
        <div className="whitespace-pre-wrap">{reply}</div>
      ) : (
        isStreaming && <PulseDots />
      )}
      {(isEmpty || hasToolMarkers) && !isStreaming && (
        <div className="mt-2 text-[10px] text-neutral-300 font-mono break-all border-t border-neutral-100 pt-2">
          raw: {JSON.stringify(content).slice(0, 500)}
        </div>
      )}
    </div>
  );
}

function AudioMessage({ waveform, sampleRate }) {
  const [playing, setPlaying] = useState(false);
  const audioCtxRef = useRef(null);

  const play = useCallback(() => {
    if (playing || !waveform) return;
    setPlaying(true);
    try {
      const ctx = audioCtxRef.current || new AudioContext({ sampleRate: sampleRate || 24000 });
      audioCtxRef.current = ctx;
      const buffer = ctx.createBuffer(1, waveform.length, sampleRate || 24000);
      buffer.getChannelData(0).set(waveform);
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      source.onended = () => setPlaying(false);
      source.start();
    } catch (err) {
      console.error("Audio playback error:", err);
      setPlaying(false);
    }
  }, [waveform, sampleRate, playing]);

  const duration = waveform ? (waveform.length / (sampleRate || 24000)).toFixed(1) : 0;

  return (
    <div className="message message-audio">
      <button
        onClick={play}
        disabled={playing || !waveform}
        className="inline-flex items-center gap-2 px-4 py-2 bg-neutral-800 text-white rounded-lg text-sm font-medium hover:bg-neutral-700 transition-colors disabled:opacity-50"
      >
        {playing ? (
          <>
            <span className="inline-block w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />
            Playing...
          </>
        ) : (
          <>
            <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
              <path d="M3 2.5v9l8-4.5z" />
            </svg>
            Play Audio ({duration}s)
          </>
        )}
      </button>
    </div>
  );
}

function DownloadIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" className="inline-block ml-1 align-[-1px]">
      <path d="M6 2v6M3.5 5.5L6 8l2.5-2.5M2.5 10h7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function ModelPills({ activeId, cachedModels, onSelect, disabled }) {
  return (
    <div className="flex flex-wrap gap-2 justify-center">
      {MODELS.map((m) => {
        const isActive = m.id === activeId;
        const isCached = cachedModels[m.id];

        let cls = "model-pill";
        if (isActive) cls += " model-pill-active";
        else if (isCached) cls += " model-pill-cached";
        else cls += " model-pill-download";

        return (
          <button
            key={m.id}
            onClick={() => onSelect(m.id)}
            disabled={disabled || isActive}
            className={cls}
            title={m.desc}
          >
            {m.label}
            {!isActive && !isCached && <DownloadIcon />}
          </button>
        );
      })}
    </div>
  );
}

// Create worker for the given model — Vite requires static new URL() paths
function createModelWorker(modelId) {
  const model = MODELS.find((m) => m.id === modelId);
  switch (model?.type) {
    case "vl":
      return new Worker(new URL("./worker-vl.js", import.meta.url), { type: "module" });
    case "audio":
      return new Worker(new URL("./worker-audio.js", import.meta.url), { type: "module" });
    default:
      return new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  }
}

function checkWebGPU() {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

function isMobile() {
  return /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
}

export default function App() {
  const [modelId, setModelId] = useState(getSelectedModel);
  const [status, setStatus] = useState("checking"); // checking | idle | loading | ready | error
  const [statusText, setStatusText] = useState("");
  const [progress, setProgress] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [generating, setGenerating] = useState(false);
  const [stats, setStats] = useState(null);
  const [cachedModels, setCachedModels] = useState({});

  const workerRef = useRef(null);
  const chatEndRef = useRef(null);
  const inputRef = useRef(null);
  const streamRef = useRef("");

  const modelInfo = getModelInfo(modelId);

  // Check cache status for all models on mount and after loading completes
  useEffect(() => {
    checkAllCaches().then(setCachedModels);
  }, [status]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleWorkerMessage = useCallback((e) => {
    const { type, data } = e.data;

    switch (type) {
      case "status":
        setStatusText(data);
        break;
      case "progress":
        if (data.status === "progress") {
          setProgress({ file: data.file, loaded: data.loaded, total: data.total });
        }
        break;
      case "loaded":
        setStatus("ready");
        setStatusText("");
        setProgress(null);
        break;
      case "generate_start":
        streamRef.current = "";
        break;
      case "token": {
        streamRef.current += data.text;
        const text = streamRef.current;
        // If audio started, show a generating indicator
        if (data.audioStarted) {
          setMessages((prev) => [
            ...prev,
            { role: "audio-pending", content: "Generating speech..." },
          ]);
          break;
        }
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last && last.role === "assistant") {
            updated[updated.length - 1] = { ...last, content: text };
          } else {
            updated.push({ role: "assistant", content: text });
          }
          return updated;
        });
        setStats({ tokensPerSec: data.tokensPerSec, tokenCount: data.tokenCount });
        break;
      }
      case "audio_data":
        // Replace audio-pending message with playable audio
        setMessages((prev) => {
          const updated = prev.filter((m) => m.role !== "audio-pending");
          return [
            ...updated,
            { role: "audio", waveform: new Float32Array(data.waveform), sampleRate: data.sampleRate },
          ];
        });
        break;
      case "tool_call":
        setMessages((prev) => [
          ...prev,
          { role: "tool", content: `Called ${data.call.name}(${Object.entries(data.call.args).map(([k, v]) => `${k}="${v}"`).join(", ")})`, result: data.result },
        ]);
        break;
      case "tool_continue":
        streamRef.current = "";
        break;
      case "generate_done":
        setGenerating(false);
        setStats({
          tokensPerSec: data.tokensPerSec,
          tokenCount: data.tokenCount,
          elapsed: data.elapsed,
        });
        // Remove any leftover audio-pending messages
        setMessages((prev) => prev.filter((m) => m.role !== "audio-pending"));
        setTimeout(() => inputRef.current?.focus(), 100);
        break;
      case "error":
        setStatus("error");
        setStatusText(data);
        setGenerating(false);
        break;
    }
  }, [modelId]);

  // Create worker and load/check model
  useEffect(() => {
    if (isMobile()) {
      setStatus("idle");
      return;
    }

    if (!checkWebGPU()) {
      setStatus("error");
      setStatusText("WebGPU is not supported in this browser. Try Chrome 113+ or Edge 113+.");
      return;
    }

    // Check if selected model is cached, then decide to auto-load or show download
    let cancelled = false;
    isModelCached(modelId).then((cached) => {
      if (cancelled) return;

      const worker = createModelWorker(modelId);
      worker.addEventListener("message", handleWorkerMessage);
      workerRef.current = worker;

      if (cached) {
        setStatus("loading");
        setStatusText("Loading from cache...");
        worker.postMessage({ type: "load", data: { modelId } });
      } else {
        setStatus("idle");
      }
    });

    return () => {
      cancelled = true;
      if (workerRef.current) {
        workerRef.current.removeEventListener("message", handleWorkerMessage);
        workerRef.current.terminate();
        workerRef.current = null;
      }
    };
  }, [handleWorkerMessage, modelId]);

  const loadModel = useCallback(() => {
    setStatus("loading");
    setStatusText("Downloading model...");
    // Worker might not exist yet if we went straight to idle
    if (!workerRef.current) {
      const worker = createModelWorker(modelId);
      worker.addEventListener("message", handleWorkerMessage);
      workerRef.current = worker;
    }
    workerRef.current.postMessage({ type: "load", data: { modelId } });
  }, [modelId, handleWorkerMessage]);

  const switchModel = useCallback((newId) => {
    if (newId === modelId) return;
    setSelectedModel(newId);
    setModelId(newId);
    setMessages([]);
    setStats(null);
    setStatus("checking");
  }, [modelId]);

  const sendMessage = useCallback(() => {
    const text = input.trim();
    if (!text || generating || status !== "ready") return;

    const newMessages = [...messages, { role: "user", content: text }];
    setMessages(newMessages);
    setInput("");
    setGenerating(true);
    setStats(null);

    // Only send standard chat roles to the worker (filter out audio/audio-pending)
    workerRef.current?.postMessage({
      type: "generate",
      data: {
        messages: newMessages
          .filter((m) => ["user", "assistant", "system", "tool"].includes(m.role))
          .map(({ role, content }) => ({ role, content })),
        maxTokens: 2048,
        temperature: 0.7,
      },
    });
  }, [input, generating, status, messages]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const hasWebGPU = checkWebGPU();
  const mobile = isMobile();

  return (
    <div className="page">
      <LabHeader />

      <section className="hero pt-24 pb-4">
        <p className="eyebrow">In-Browser AI</p>
        <h1 className="title">LiquidAI LFM 2.5 WebGPU</h1>
        <p className="lede">
          Run Liquid AI's LFM 2.5 models entirely in your browser. No server, no API keys — just WebGPU.
        </p>
      </section>

      <section className="content">
        {/* Mobile block */}
        {mobile && (
          <div className="panel text-center space-y-3">
            <p className="text-lg font-semibold">Desktop Required</p>
            <p className="text-sm text-neutral-500">
              Loading a 1B+ parameter model needs more memory than mobile browsers can handle.
              Try this on a desktop with Chrome or Edge.
            </p>
          </div>
        )}

        {!mobile && (
          <div className="space-y-4">
            {/* Model pills — always visible */}
            <ModelPills
              activeId={status === "ready" ? modelId : null}
              cachedModels={cachedModels}
              onSelect={switchModel}
              disabled={generating || status === "loading" || status === "checking"}
            />

            {/* Checking cache */}
            {status === "checking" && (
              <div className="panel text-center">
                <span className="text-sm text-neutral-500">Checking for cached model</span>
                <PulseDots />
              </div>
            )}

            {/* Download prompt for uncached model */}
            {status === "idle" && (
              <div className="panel text-center space-y-4">
                <p className="text-sm text-neutral-500">
                  {hasWebGPU ? (
                    <>Ready to download <strong>{modelInfo.label}</strong> ({modelInfo.size}).</>
                  ) : (
                    <span className="text-red-600">
                      WebGPU not available. Use Chrome 113+ or Edge 113+.
                    </span>
                  )}
                </p>
                <button
                  onClick={loadModel}
                  disabled={!hasWebGPU}
                  className="px-6 py-3 bg-neutral-800 text-white rounded-xl font-medium hover:bg-neutral-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  Download & Load {modelInfo.label}
                </button>
                <p className="text-xs text-neutral-400">
                  Model weights are cached in your browser after first download.
                </p>
              </div>
            )}

            {/* Loading */}
            {status === "loading" && (
              <div className="panel space-y-4">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium">{statusText}</span>
                  <PulseDots />
                </div>
                <ProgressBar progress={progress} />
              </div>
            )}

            {/* Error */}
            {status === "error" && (
              <div className="panel border-red-200 bg-red-50/80 space-y-3">
                <p className="text-sm text-red-700 font-medium">Something went wrong</p>
                <p className="text-sm text-red-600 font-mono break-all">{statusText}</p>
                <button
                  onClick={() => { setStatus("idle"); setStatusText(""); }}
                  className="text-sm text-red-600 underline hover:text-red-800"
                >
                  Try again
                </button>
              </div>
            )}

            {/* Chat Interface */}
            {(status === "ready" || (status === "loading" && messages.length > 0)) && (
              <div className="space-y-4">
                {/* Stats */}
                {stats && (
                  <div className="flex justify-center gap-2">
                    <span className="stats-badge">
                      {stats.tokensPerSec?.toFixed(1)} tok/s
                    </span>
                    {stats.tokenCount && (
                      <span className="stats-badge">
                        {stats.tokenCount} tokens
                      </span>
                    )}
                    {stats.elapsed && (
                      <span className="stats-badge">
                        {stats.elapsed.toFixed(1)}s
                      </span>
                    )}
                  </div>
                )}

                {/* Messages */}
                <div className="chat-container min-h-[200px] max-h-[60vh] overflow-y-auto panel">
                  {messages.length === 0 && (
                    <p className="text-sm text-neutral-400 text-center py-8">
                      Model loaded. Type something to start chatting.
                    </p>
                  )}
                  {messages.map((msg, i) =>
                    msg.role === "user" ? (
                      <div key={i} className="message message-user">
                        <div className="whitespace-pre-wrap">{msg.content}</div>
                      </div>
                    ) : msg.role === "tool" ? (
                      <div key={i} className="message message-tool">
                        <span className="tool-label">tool</span> {msg.content}
                      </div>
                    ) : msg.role === "audio" ? (
                      <AudioMessage key={i} waveform={msg.waveform} sampleRate={msg.sampleRate} />
                    ) : msg.role === "audio-pending" ? (
                      <div key={i} className="message message-assistant">
                        <span className="text-neutral-400 text-xs font-medium">Generating speech</span>
                        <PulseDots />
                      </div>
                    ) : (
                      <AssistantMessage
                        key={i}
                        content={msg.content}
                        isStreaming={generating && i === messages.length - 1}
                      />
                    )
                  )}
                  {generating && messages[messages.length - 1]?.role !== "assistant" && (
                    <div className="message message-assistant">
                      <PulseDots />
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>

                {/* Input */}
                <div className="flex gap-3">
                  <textarea
                    ref={inputRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Type a message..."
                    disabled={generating}
                    rows={1}
                    className="flex-1 px-5 py-3.5 bg-white border border-neutral-200 rounded-xl resize-none focus:outline-none focus:border-neutral-400 transition-colors disabled:opacity-50 text-sm"
                  />
                  <button
                    onClick={sendMessage}
                    disabled={generating || !input.trim()}
                    className="px-5 py-3.5 bg-neutral-800 text-white rounded-xl font-medium hover:bg-neutral-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed text-sm"
                  >
                    {generating ? "..." : "Send"}
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </section>
    </div>
  );
}

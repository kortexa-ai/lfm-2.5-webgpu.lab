import { useState, useRef, useEffect, useCallback } from "react";

const SLUG = "lfm-2-5-webgpu";

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

// Parse <think>...</think> blocks from assistant content.
// Handles streaming where tags arrive token-by-token.
function parseThinking(content) {
  // Still assembling the opening tag (e.g. "<", "<t", "<thi"...)
  if (!content.includes("<think>") && "<think>".startsWith(content.trim())) {
    return { thinking: null, reply: "", thinkingDone: false };
  }

  if (!content.includes("<think>")) {
    return { thinking: null, reply: cleanSpecialTokens(content), thinkingDone: true };
  }

  const afterOpen = content.split("<think>")[1];

  // </think> is present — split thinking from reply
  if (afterOpen.includes("</think>")) {
    const [thinking, ...rest] = afterOpen.split("</think>");
    return { thinking: cleanSpecialTokens(thinking.trim()), reply: cleanSpecialTokens(rest.join("</think>").trim()), thinkingDone: true };
  }

  // Still streaming thinking content
  return { thinking: cleanSpecialTokens(afterOpen.trim()), reply: "", thinkingDone: false };
}

// Strip tool call markers and other special tokens from displayed text
function cleanSpecialTokens(text) {
  return text
    .replace(/<\|tool_call_start\|>[\s\S]*?<\|tool_call_end\|>/g, "")
    .replace(/<\|tool_call_start\|>[\s\S]*/g, "") // partial, still streaming
    .replace(/<\|[a-z_]+\|>/g, "") // any remaining special tokens like <|im_end|>
    .replace(/◁[a-z_]+▷/g, "") // alternate rendering of special tokens
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

  // While <think> is streaming: just show "Thinking..." pulse
  if (stillThinking) {
    return (
      <div className="message message-assistant">
        <span className="text-neutral-400 text-xs font-medium">Thinking</span>
        <PulseDots />
      </div>
    );
  }

  // Debug: detect if content has tool markers or is suspiciously empty
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

function checkWebGPU() {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

function isMobile() {
  return /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
}

export default function App() {
  const [status, setStatus] = useState("checking"); // checking | idle | loading | ready | error
  const [statusText, setStatusText] = useState("");
  const [progress, setProgress] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [generating, setGenerating] = useState(false);
  const [stats, setStats] = useState(null);

  const workerRef = useRef(null);
  const chatEndRef = useRef(null);
  const inputRef = useRef(null);
  // Track the assistant reply being streamed
  const streamRef = useRef("");

  // Scroll chat to bottom
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
      case "tool_call":
        // Show tool execution in chat
        setMessages((prev) => [
          ...prev,
          { role: "tool", content: `Called ${data.call.name}(${Object.entries(data.call.args).map(([k, v]) => `${k}="${v}"`).join(", ")})`, result: data.result },
        ]);
        break;
      case "tool_continue":
        // Reset stream for the continuation after tool result
        streamRef.current = "";
        break;
      case "generate_done":
        setGenerating(false);
        setStats({
          tokensPerSec: data.tokensPerSec,
          tokenCount: data.tokenCount,
          elapsed: data.elapsed,
        });
        // Focus input after generation
        setTimeout(() => inputRef.current?.focus(), 100);
        break;
      case "error":
        setStatus("error");
        setStatusText(data);
        setGenerating(false);
        break;
      case "cache_status":
        if (data.cached) {
          // Cached — auto-load immediately
          setStatus("loading");
          setStatusText("Loading from cache...");
          workerRef.current?.postMessage({ type: "load", data: {} });
        } else {
          setStatus("idle");
        }
        break;
    }
  }, []);

  // Create worker on mount and check cache
  useEffect(() => {
    if (!checkWebGPU()) {
      setStatus("error");
      setStatusText("WebGPU is not supported in this browser. Try Chrome 113+ or Edge 113+.");
      return;
    }

    const worker = new Worker(new URL("./worker.js", import.meta.url), {
      type: "module",
    });
    worker.addEventListener("message", handleWorkerMessage);
    workerRef.current = worker;

    worker.postMessage({ type: "check", data: {} });
  }, [handleWorkerMessage]);

  const loadModel = useCallback(() => {
    setStatus("loading");
    setStatusText("Initializing...");
    workerRef.current?.postMessage({ type: "load", data: {} });
  }, []);

  const sendMessage = useCallback(() => {
    const text = input.trim();
    if (!text || generating || status !== "ready") return;

    const newMessages = [...messages, { role: "user", content: text }];
    setMessages(newMessages);
    setInput("");
    setGenerating(true);
    setStats(null);

    workerRef.current?.postMessage({
      type: "generate",
      data: {
        messages: newMessages.map(({ role, content }) => ({ role, content })),
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
          Run Liquid AI's LFM 2.5 1.2B Thinking entirely in your browser. No server, no API keys — just WebGPU.
        </p>
      </section>

      <section className="content">
        {/* Mobile block */}
        {mobile && (
          <div className="panel text-center space-y-3">
            <p className="text-lg font-semibold">Desktop Required</p>
            <p className="text-sm text-neutral-500">
              Loading a 1.2B parameter model needs more memory than mobile browsers can handle.
              Try this on a desktop with Chrome or Edge.
            </p>
          </div>
        )}

        {/* Checking cache */}
        {!mobile && status === "checking" && (
          <div className="panel text-center">
            <span className="text-sm text-neutral-500">Checking for cached model</span>
            <PulseDots />
          </div>
        )}

        {/* Status / Loader */}
        {status === "idle" && (
          <div className="panel text-center space-y-4">
            <div className="space-y-2">
              <p className="text-sm text-neutral-500">
                {hasWebGPU ? (
                  <>WebGPU detected. Ready to load model (~600MB download).</>
                ) : (
                  <span className="text-red-600">
                    WebGPU not available. Use Chrome 113+ or Edge 113+.
                  </span>
                )}
              </p>
            </div>
            <button
              onClick={loadModel}
              disabled={!hasWebGPU}
              className="px-6 py-3 bg-neutral-800 text-white rounded-xl font-medium hover:bg-neutral-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              Load LFM 2.5 1.2B
            </button>
            <p className="text-xs text-neutral-400">
              Model weights are cached in your browser after first download.
            </p>
          </div>
        )}

        {status === "loading" && (
          <div className="panel space-y-4">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium">{statusText}</span>
              <PulseDots />
            </div>
            <ProgressBar progress={progress} />
          </div>
        )}

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
            {/* Stats bar */}
            {stats && (
              <div className="flex justify-center gap-3">
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
      </section>
    </div>
  );
}

# Reachy Office Manager

A real-time voice AI agent, embodied as a **Reachy Mini** desk robot, acting as the
**valantic Office Manager**: it sees the room, talks to you, manages office desk/meeting-room
bookings, remembers things about you across conversations, and can look things up on the web.

Built on the **NVIDIA NeMo Agent Toolkit** (tool-calling agent + LLM router) and
**pipecat** (real-time voice/video pipeline), with a custom **office availability/booking
MCP server** and a live web dashboard.

## What it does

- **Voice conversation** over WebRTC (browser mic/speaker), German by default.
- **Vision on demand**: describes what it sees, counts people in the room, or does a
  "look around" scan (turns left/front/right, remembers what it saw for the rest of the
  conversation) — never a stale/automatic camera attach, only when the agent actually needs to look.
- **Office management**: checks and books free desks/meeting rooms live via MCP tools,
  gives spoken directions to the room, re-checks availability before confirming a booking.
- **Physical personality**: gestures and emotions (used sparingly, real moments only),
  face tracking, natural idle breathing/sway.
- **Persistent memory** across all conversations (SQLite-backed) — remembers stated
  preferences and recalls them instead of asking twice.
- **Web search** for anything current it can't know from training (cafeteria menu,
  restaurants, opening hours, ...).
- **Filler phrases** ("Einen Moment, ich schaue nach...") to bridge latency on slow
  tool calls (MCP/vision), never on quick commands like turning.

## Architecture

Four services run together:

| Service  | What                                                              | Port |
|----------|-------------------------------------------------------------------|------|
| `bot`    | pipecat voice/video pipeline, robot control, WebRTC frontend      | 7860 (WebRTC), 7861 (internal control API) |
| `nat`    | NeMo Agent Toolkit: LLM router + tool-calling agent                | 8001 |
| `mcp`    | Office availability/booking MCP server (FastMCP)                   | 8002 |
| `webapp` | Live office-occupancy dashboard (reads/writes the same booking DB) | 8003 |

```
                     ┌─────────────┐
   Browser  ◄──WebRTC──►   bot      │──HTTP (control API)──►  Reachy Mini daemon
  (mic/cam/speaker)   └──────┬──────┘
                             │ HTTP (chat completions)
                             ▼
                      ┌─────────────┐        ┌──────────────┐
                      │     nat     │◄──MCP──►│     mcp      │◄──┐
                      │ router+agent│        │ (booking DB) │   │ shared
                      └─────────────┘        └──────────────┘   │ SQLite
                                                     ▲            │
                                              ┌──────┴──────┐    │
                                              │   webapp    │────┘
                                              │ (dashboard) │
                                              └─────────────┘
```

The NAT agent has its own persistent memory (`nat/src/ces_tutorial/agent_memory.db`,
separate from the office booking DB) and calls out to web search independently.

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- API keys: OpenAI (STT/TTS/vision/chat), NVIDIA (for `nat`), ElevenLabs (optional TTS voice), HF token

## Setup

### 1. Create the environment file

Create a `.env` in the repo root:

```bash
OPENAI_API_KEY=...
NVIDIA_API_KEY=...
ELEVENLABS_API_KEY=...
HF_TOKEN=...
REACHY_USE_SIM=true        # or false for real hardware
REACHY_ROBOT_NAME=reachy_mini
REACHY_HOST=...            # only needed for real hardware
```

### 2. Install each service's dependencies

```bash
cd bot && uv sync && cd ..
cd nat && uv sync && cd ..
cd office-mcp && uv sync && cd ..
```

### 3. Start the Reachy Mini daemon (separate from the four services below)

**macOS:**
```bash
cd bot
uv run mjpython -m reachy_mini.daemon.app.main --sim --no-localhost-only
```

**Linux:**
```bash
cd bot
uv run -m reachy_mini.daemon.app.main --sim --no-localhost-only
```

Drop `--sim` to run on real hardware.

## Running everything else

A centralized CLI starts/stops/restarts all four services (`bot`, `nat`, `mcp`, `webapp`) in
one terminal, with combined, color-tagged, filtered logs:

```bash
# one-time: add scripts/ to your PATH (see scripts/dev.py header / ask your shell rc)
reachy-start              # start everything, Ctrl+C to stop
reachy-start bot nat      # start only specific services
reachy-stop               # kill whatever's on the known ports
reachy-restart            # stop then start
reachy-status             # show which services are up
```

Or directly, without the PATH setup:

```bash
python3 scripts/dev.py start
```

Then open **http://localhost:7860/client** in a browser for the voice/video call, and
**http://localhost:8003** for the office occupancy dashboard.

Console log level defaults to `INFO` (quiet); set `LOG_LEVEL=DEBUG` before starting to see
full per-frame pipeline tracing.

## Project Structure

```
reachy-office-manager/
├── bot/                        # pipecat voice/video pipeline + robot control
│   ├── main.py                 # pipeline assembly, WebRTC transport, greeting
│   ├── nat_vision_llm.py       # routes chat + on-demand vision to the nat agent
│   └── services/
│       ├── reachy_service.py   # robot motion/camera/gesture control
│       ├── control_api.py      # HTTP bridge so `nat` (separate process) can drive the robot
│       ├── processor.py        # filler-phrase / wobble frame processor
│       └── ...
├── nat/                         # NeMo Agent Toolkit: router + tool-calling agent
│   └── src/ces_tutorial/
│       ├── config.yml           # LLM, tools, system prompt, routing config
│       ├── memory_store.py      # persistent cross-conversation memory (SQLite)
│       └── functions/           # reachy_* tools, web_search, remember/recall, MCP wrappers
├── office-mcp/                  # office availability/booking MCP server + dashboard
│   ├── mcp_server.py            # FastMCP tools (list/book/release desks & rooms)
│   ├── office_layout.py         # room/desk layout definitions
│   ├── store.py                 # booking state (SQLite)
│   ├── webapp.py                # dashboard backend
│   └── frontend/                # dashboard UI (floor plan, live activity feed)
├── scripts/
│   ├── dev.py                   # centralized launcher (start/stop/status, colored logs)
│   └── reachy, reachy-start/stop/restart/status   # thin CLI wrappers around dev.py
└── .env                         # API keys (create this, gitignored)
```

## Troubleshooting

- **Port conflicts**: `reachy-status` shows what's up; `reachy-stop` frees all four ports.
- **API key errors**: verify `.env` is in the repo root and has valid keys.
- **Robot connection issues**: make sure the Reachy Mini daemon is running before `bot`.
- **Noisy logs**: default level is `INFO` on purpose; `LOG_LEVEL=DEBUG reachy-start` for full tracing.
- **MCP tool name errors**: tools are wired via `mcp_tool_wrapper` (not `mcp_client`) in
  `nat/src/ces_tutorial/config.yml` — the latter prefixes tool names with `office_tools.`,
  which OpenAI's tool-calling API rejects.

## Resources

- [NVIDIA NeMo Agent Toolkit](https://github.com/NVIDIA/NeMo-Agent-Toolkit)
- [Reachy Mini Robot](https://www.pollen-robotics.com/)
- [pipecat](https://github.com/pipecat-ai/pipecat)

## ARIS QA Plan

This file defines the high-level QA plan for ARIS. It is intended as a release and regression checklist covering the full pipeline: **audio → transcripts → callsigns → QSOs → web UI and control APIs**.

### 1. Core Pipeline (Audio → Transcript → Callsign → QSO)

- **CP-1: Single-slot happy path (USB voice)**
  - **Preconditions**: Docker stack running (`docker compose up -d`), at least one RX slot configured to a known voice frequency.
  - **Steps**:
    - Start slot 1 via Web UI or `POST /api/slots/1/start`.
    - Wait 1–2 minutes with real or mock audio.
    - Call `GET /api/stats`, `GET /api/transcripts?limit=10`, `GET /api/callsigns?limit=10`, `GET /api/qsos?limit=5`.
  - **Expected**:
    - `audio_flowing` is `true`.
    - New transcripts appear with reasonable text and confidence.
    - Callsigns show up (when present in audio).
    - At least one QSO is present after sustained activity.

- **CP-2: Timestamp/frequency consistency**
  - **Steps**:
    - Capture timestamps and `frequency_hz` from recent audio (via `/api/stats` and `/api/audio/latest`) and from `/api/transcripts`, `/api/callsigns`, `/api/qsos`.
  - **Expected**:
    - For a given time window, transcripts and callsigns share the same `frequency_hz` as the underlying audio.
    - QSO entries reference the same `frequency_hz` and a time range that aligns with underlying transcripts.

- **CP-3: Persistence across restarts**
  - **Steps**:
    - With active data flowing, record counts from `/api/stats` and samples from `/api/transcripts` and `/api/qsos`.
    - Restart `api` and `summarizer` services.
    - Re-query the same endpoints.
  - **Expected**:
    - Historical transcripts and QSOs remain present (SQLite persistence).
    - New data continues to append after restart.

### 2. Multi-Slot Functionality

- **MS-1: Parallel slots**
  - **Preconditions**: At least two valid KiwiSDR endpoints or frequencies.
  - **Steps**:
    - Start slots 1–2 (or up to 4) with distinct `frequency_hz` and/or hosts via Web UI or `/api/slots/{id}/start`.
    - Let them run for several minutes.
  - **Expected**:
    - `/api/slots` shows each slot as `online` with the configured host, port, frequency, and mode.
    - Web Dashboard shows each `RX_SLOT_n` as `LINK_ACTIVE` with matching frequency/mode.
    - `/api/transcripts` contains interleaved transcripts from all active frequencies.

- **MS-2: Slot isolation**
  - **Steps**:
    - With multiple active slots, stop a single slot via Web UI or `POST /api/slots/{id}/stop`.
  - **Expected**:
    - Only that slot moves to `offline` in `/api/slots` and UI.
    - Remaining slots continue to produce transcripts/callsigns/QSOs without interruption.

- **MS-3: Start/stop churn**
  - **Steps**:
    - Rapidly start/stop a slot several times in a row.
  - **Expected**:
    - No duplicate or “ghost” slots in `/api/slots`.
    - Dashboard state remains in sync; no stuck `LINK_ACTIVE` when audio is clearly stopped.

### 3. Modes and Sidebands (USB / LSB / AM / FM / CW)

- **MD-1: Voice modes (USB / LSB / AM / FM)**
  - **Steps**:
    - For each mode, configure a slot with known voice content.
    - Let it run long enough to collect multiple transcripts.
  - **Expected**:
    - Transcripts are intelligible (no persistent noise-only output).
    - Confidence values are in a reasonable range (not always near 0 or 100% with garbage text).

- **MD-2: CW mode**
  - **Steps**:
    - Configure one slot in `CW` mode with real or simulated CW audio.
  - **Expected**:
    - `STREAM_TRANSCRIPTS` produces alphanumeric CW-decoded text (no persistent empty or punctuation-only output).
    - Web transcript feed shows `mode: "CW"` for these entries.

- **MD-3: Mode switching**
  - **Steps**:
    - While a slot is active, change its mode (e.g., USB → CW → USB) via control endpoint or by editing slot config and restarting the slot.
  - **Expected**:
    - Audio and transcripts resume under the new mode without crashes.
    - CW-specific behavior is used only when `mode == "CW"`.

### 4. Web Frontend (Dashboard, Slots, Transcript Feed)

- **UI-1: Dashboard initial load**
  - **Steps**:
    - With at least one active slot and transcripts flowing, open `http://localhost:3000`.
  - **Expected**:
    - `RX_SLOT_n` cards reflect current live status and configuration (host/port/frequency/mode) without extra user interaction.
    - Top-level stats (WS status, counts) appear when WebSocket is connected.

- **UI-2: Slot control from UI**
  - **Steps**:
    - Use `INITIALIZE LINK` and `TERMINATE LINK` buttons on each `RX_SLOT_n`.
  - **Expected**:
    - Corresponding `/api/slots/{id}/start` or `/api/slots/{id}/stop` behavior is observed.
    - Slot status in the Dashboard matches `/api/slots`.

- **UI-3: Transcript feed real-time streaming**
  - **Steps**:
    - With audio flowing, open the Dashboard and watch `INTELLIGENCE_FEED`.
    - Observe for several minutes without refreshing.
  - **Expected**:
    - New transcripts appear automatically via WebSocket (no page refresh required).
    - BUFFER count reflects number of loaded items (typically capped at ~50).

- **UI-4: Transcript deletion**
  - **Steps**:
    - Click delete on a transcript in the feed.
  - **Expected**:
    - Transcript disappears from the feed.
    - `GET /api/transcripts` no longer returns that transcript ID.

### 5. LLM Summaries and QSO Handling

- **LLM-1: QSO creation**
  - **Steps**:
    - Run one or more voice slots over a real QSO or test conversation.
    - After a suitable period, query `GET /api/qsos?limit=10` and view QSO list in UI.
  - **Expected**:
    - New QSO entries appear with plausible `session_id`, `start_time`, `end_time`, `frequency_hz`, `mode`, and `callsigns`.

- **LLM-2: Summary quality**
  - **Steps**:
    - Manually read several QSO summaries in UI and via `/api/qsos`.
  - **Expected**:
    - Summaries roughly match conversation topic and participants.
    - No obvious hallucinations or completely unrelated content.

- **LLM-3: Manual summarization trigger**
  - **Steps**:
    - Use Web UI “Summarize” button or `POST /api/control/summarize`.
  - **Expected**:
    - New or updated QSO summaries appear within a short delay.

- **LLM-4: Regenerate summary**
  - **Steps**:
    - Call `POST /api/qsos/{session_id}/regenerate` for an existing QSO.
  - **Expected**:
    - Corresponding summary updates; no duplicate QSO rows are created.

- **LLM-5: QSO deletion**
  - **Steps**:
    - Delete a QSO via Web UI or `DELETE /api/qsos/{session_id}`.
  - **Expected**:
    - QSO is removed from `/api/qsos` and UI.
    - Associated transcripts are deleted if that behavior is configured (current implementation deletes transcripts in that time window).

### 6. Control APIs (Frequency, Mode, Filters, AGC, Noise Blanker)

- **CTL-1: Frequency control**
  - **Steps**:
    - Call `POST /api/control/frequency` with a new `frequency_khz`.
    - Wait a few seconds and query `/api/stats` and `/api/slots`.
  - **Expected**:
    - Recent audio in stats reflects the new frequency.
    - Affected slots show updated `frequency_hz` in UI/slots endpoint.

- **CTL-2: Mode control**
  - **Steps**:
    - Call `POST /api/control/mode` with a new mode (USB/LSB/AM/FM/CW).
  - **Expected**:
    - `recent_audio.mode` updates in `/api/stats`.
    - Corresponding slot mode updates in UI.

- **CTL-3: Filter / AGC / Noise blanker**
  - **Steps**:
    - Call `POST /api/control/filter`, `/api/control/agc`, `/api/control/noise-blanker` with valid parameters.
  - **Expected**:
    - 2xx response codes.
    - Audio behavior matches expectations (no regressions like total silence or clipping), verified by listening to `/api/audio/latest`.

### 7. Resilience and Failure Modes

- **RS-1: Service restarts**
  - **Steps**:
    - With active traffic, restart each service one at a time (`audio-capture`, `stt`, `callsign-extractor`, `summarizer`, `api`, `web`).
  - **Expected**:
    - Pipeline resumes after each restart.
    - UI recovers WebSocket connection (`WS_CONNECTED` state) and resumes streaming.

- **RS-2: Redis restart**
  - **Steps**:
    - Briefly stop Redis and bring it back, then observe logs and behavior.
  - **Expected**:
    - Services reconnect without manual intervention.
    - No unbounded error spam in logs after Redis returns.

- **RS-3: Browser / network blips**
  - **Steps**:
    - Disconnect and reconnect the browser (or simulate network loss).
  - **Expected**:
    - WebSocket reconnect logic runs and UI transitions back to `WS_CONNECTED`.
    - New transcripts and slot updates resume streaming.

### 8. Performance / Load (Optional but Recommended)

- **PF-1: Multi-slot sustained load**
  - **Steps**:
    - Run 3–4 slots for several hours under typical band activity.
    - Monitor `docker compose logs` and basic system metrics (CPU, RAM, GPU usage).
  - **Expected**:
    - No crashes or memory leaks observed.
    - Counts in `/api/stats` grow steadily without errors.

- **PF-2: Backlog processing**
  - **Steps**:
    - Generate a backlog of audio in Redis (e.g., pause STT briefly, then resume).
  - **Expected**:
    - STT service processes pending audio and catches up.
    - No permanent “stuck” messages in the Redis streams consumer group.


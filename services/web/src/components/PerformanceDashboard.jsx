import React, { useEffect, useState, useMemo } from 'react';

const API_BASE = import.meta.env.VITE_API_URL || '';

// Simple in-memory limit for history points
const MAX_POINTS = 120; // e.g. ~10 minutes at 5s interval

const PerformanceDashboard = () => {
  const [samples, setSamples] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    const fetchStats = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/stats`, {
          credentials: 'include',
        });
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const data = await res.json();
        if (cancelled) return;

        const now = Date.now();
        const stt = data.stt_metrics || {};
        const summarizer = data.summarizer_metrics || {};

        const point = {
          t: now,
          audio_chunks_count: data.audio_chunks_count,
          transcripts_count: data.transcripts_count,
          qsos_count: data.qsos_count,
          stt_latency_ms: stt.last_latency_ms ?? null,
          stt_buffer_ms: stt.last_buffer_ms ?? null,
          summarizer_latency_ms: summarizer.last_latency_ms ?? null,
        };

        setSamples((prev) => {
          const next = [...prev, point];
          if (next.length > MAX_POINTS) {
            next.shift();
          }
          return next;
        });
        setError(null);
      } catch (e) {
        if (!cancelled) {
          setError(e.message || 'Failed to load stats');
        }
      }
    };

    // Initial fetch
    fetchStats();
    // Poll every 5 seconds
    const id = setInterval(fetchStats, 5000);

    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const latest = samples[samples.length - 1] || {};

  // Series are kept in seconds for display
  const sttSeries = useMemo(
    () =>
      samples
        .filter((p) => typeof p.stt_latency_ms === 'number' && !Number.isNaN(p.stt_latency_ms))
        .map((p) => ({ t: p.t, v: p.stt_latency_ms / 1000 })),
    [samples]
  );

  const summarizerSeries = useMemo(
    () =>
      samples
        .filter(
          (p) =>
            typeof p.summarizer_latency_ms === 'number' &&
            !Number.isNaN(p.summarizer_latency_ms)
        )
        .map((p) => ({ t: p.t, v: p.summarizer_latency_ms / 1000 })),
    [samples]
  );

  const describeSeries = (series) => {
    if (!series || series.length === 0) {
      return 'NO DATA';
    }
    const values = series.map((p) => p.v);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const avg = values.reduce((a, b) => a + b, 0) / values.length;
    return `N=${values.length}  MIN=${min.toFixed(2)}  AVG=${avg.toFixed(
      2
    )}  MAX=${max.toFixed(2)} s`;
  };

  const renderMiniGraph = (series, height = 60) => {
    if (!series || series.length < 2) {
      return <div style={{ color: '#555', fontSize: '0.8em' }}>INSUFFICIENT DATA</div>;
    }

    const minT = series[0].t;
    const maxT = series[series.length - 1].t || minT + 1;
    const minV = Math.min(...series.map((p) => p.v));
    const maxV = Math.max(...series.map((p) => p.v));
    const rangeV = maxV - minV || 1;

    const width = 260;

    const pointsAttr = series
      .map((p) => {
        const x = ((p.t - minT) / (maxT - minT)) * width;
        const y = height - ((p.v - minV) / rangeV) * height;
        return `${x},${y}`;
      })
      .join(' ');

    return (
      <svg width={width} height={height}>
        <polyline
          fill="none"
          stroke="var(--color-primary)"
          strokeWidth="2"
          points={pointsAttr}
        />
      </svg>
    );
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%' }}>
      <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        <div className="panel-header">
          <span>PERFORMANCE_MONITOR</span>
        </div>
        <div style={{ padding: '15px', fontSize: '0.9em' }}>
          {error && (
            <div style={{ marginBottom: '10px', color: '#ff5555' }}>
              ERROR FETCHING STATS: {error}
            </div>
          )}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
              gap: '15px',
            }}
          >
            <div className="stat-card">
              <div className="stat-label">TRANSCRIPTS_TOTAL</div>
              <div className="stat-value">{latest.transcripts_count ?? '—'}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">QSOS_TOTAL</div>
              <div className="stat-value">{latest.qsos_count ?? '—'}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">STT_LATENCY_SEC</div>
              <div className="stat-value">
                {typeof latest.stt_latency_ms === 'number'
                  ? (latest.stt_latency_ms / 1000).toFixed(2)
                  : '—'}
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-label">STT_BUFFER_SEC</div>
              <div className="stat-value">
                {typeof latest.stt_buffer_ms === 'number'
                  ? (latest.stt_buffer_ms / 1000).toFixed(2)
                  : '—'}
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-label">SUMMARIZER_LATENCY_SEC</div>
              <div className="stat-value">
                {typeof latest.summarizer_latency_ms === 'number'
                  ? (latest.summarizer_latency_ms / 1000).toFixed(2)
                  : '—'}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', flex: 1 }}>
        <div className="panel" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="panel-header">
            <span>STT_LATENCY_HISTORY_MS</span>
          </div>
          <div style={{ padding: '15px', flex: 1 }}>
            {renderMiniGraph(sttSeries)}
            <div style={{ marginTop: '8px', fontSize: '0.8em', color: '#777' }}>
              {describeSeries(sttSeries)}
            </div>
          </div>
        </div>

        <div className="panel" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="panel-header">
            <span>SUMMARIZER_LATENCY_HISTORY_MS</span>
          </div>
          <div style={{ padding: '15px', flex: 1 }}>
            {renderMiniGraph(summarizerSeries)}
            <div style={{ marginTop: '8px', fontSize: '0.8em', color: '#777' }}>
              {describeSeries(summarizerSeries)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PerformanceDashboard;


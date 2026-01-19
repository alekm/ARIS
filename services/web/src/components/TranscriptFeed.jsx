import React, { useEffect, useRef } from 'react';
import { Terminal } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || `http://${window.location.hostname}:8000`;
const TranscriptFeed = ({ transcripts }) => {
    const bottomRef = useRef(null);

    // Auto-scroll logic could go here

    return (
        <div className="panel" style={{
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden'
        }}>
            <div className="panel-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <Terminal size={14} />
                    <span>INTELLIGENCE_FEED</span>
                </div>
                <div style={{ fontSize: '0.8em', color: '#666' }}>
                    BUFFER: {transcripts.length}
                </div>
            </div>

            <div style={{ flex: 1, overflowY: 'auto', padding: '10px' }}>
                {transcripts.map((t, i) => (
                    <div key={t.id || i} style={{
                        marginBottom: '10px',
                        paddingBottom: '10px',
                        borderBottom: '1px solid #222',
                        fontFamily: 'var(--font-mono)',
                        fontSize: '0.9em'
                    }}>
                        <div style={{ display: 'flex', gap: '10px', marginBottom: '4px', color: '#666', fontSize: '0.8em' }}>
                            <span>[{t.datetime.split('T')[1].split('.')[0]}]</span>
                            <span style={{ color: 'var(--color-primary)' }}>{(t.frequency_hz / 1000).toFixed(1)}kHz</span>
                            <span>{t.mode}</span>
                            <span>CONF:{(t.confidence * 100).toFixed(0)}%</span>
                        </div>

                        <div style={{ color: '#ccc', paddingLeft: '20px', borderLeft: '2px solid #333' }}>
                            {t.text}
                        </div>
                    </div>
                ))}
                <div ref={bottomRef} />
            </div>
        </div>
    );
};

export default TranscriptFeed;

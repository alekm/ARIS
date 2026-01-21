import React, { useEffect, useRef } from 'react';
import { Terminal, Trash2 } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || '';
const TranscriptFeed = ({ transcripts, onDelete }) => {
    const bottomRef = useRef(null);

    const handleDelete = async (e, transcriptId) => {
        e.stopPropagation();
        if (!transcriptId) {
            console.warn("Cannot delete transcript without ID");
            return;
        }
        if (!window.confirm("Delete this transcript permanently?")) return;

        try {
            const res = await fetch(`${API_BASE}/api/transcripts/${transcriptId}`, { 
                method: 'DELETE',
                credentials: 'include',
            });
            if (res.ok) {
                // Call parent's onDelete callback to update the list
                if (onDelete) {
                    onDelete(transcriptId);
                }
            } else {
                console.error("Delete failed:", res.status);
                alert("Failed to delete transcript");
            }
        } catch (err) {
            console.error("Delete error:", err);
            alert("Error deleting transcript");
        }
    };

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
                {transcripts.map((t, i) => {
                    // Safely format datetime
                    let timeStr = '--:--:--';
                    if (t.datetime) {
                        try {
                            timeStr = t.datetime.split('T')[1]?.split('.')[0] || '--:--:--';
                        } catch (e) {
                            console.warn("Error parsing datetime:", t.datetime, e);
                        }
                    } else if (t.timestamp) {
                        // Fallback to timestamp if datetime not available
                        try {
                            const date = new Date(t.timestamp * 1000);
                            timeStr = date.toTimeString().split(' ')[0];
                        } catch (e) {
                            console.warn("Error formatting timestamp:", t.timestamp, e);
                        }
                    }
                    
                    return (
                        <div key={t.id || `t-${t.timestamp}-${i}`} style={{
                            marginBottom: '10px',
                            paddingBottom: '10px',
                            borderBottom: '1px solid #222',
                            fontFamily: 'var(--font-mono)',
                            fontSize: '0.9em',
                            position: 'relative'
                        }}>
                            <div style={{ display: 'flex', gap: '10px', marginBottom: '4px', color: '#666', fontSize: '0.8em', alignItems: 'center' }}>
                                <span>[{timeStr}]</span>
                                <span style={{ color: 'var(--color-primary)' }}>
                                    {t.frequency_hz ? `${(t.frequency_hz / 1000).toFixed(1)}kHz` : '--kHz'}
                                </span>
                                <span>{t.mode || '--'}</span>
                                <span>CONF:{t.confidence ? (t.confidence * 100).toFixed(0) : '--'}%</span>
                                {t.id && (
                                    <button
                                        onClick={(e) => handleDelete(e, t.id)}
                                        style={{
                                            marginLeft: 'auto',
                                            background: 'none',
                                            border: 'none',
                                            color: '#666',
                                            cursor: 'pointer',
                                            padding: '2px 4px',
                                            display: 'flex',
                                            alignItems: 'center'
                                        }}
                                        onMouseEnter={(e) => e.target.style.color = '#f44'}
                                        onMouseLeave={(e) => e.target.style.color = '#666'}
                                        title="Delete Transcript"
                                    >
                                        <Trash2 size={12} />
                                    </button>
                                )}
                            </div>

                            <div style={{ color: '#ccc', paddingLeft: '20px', borderLeft: '2px solid #333' }}>
                                {t.text || '(no text)'}
                            </div>
                        </div>
                    );
                })}
                <div ref={bottomRef} />
            </div>
        </div>
    );
};

export default TranscriptFeed;

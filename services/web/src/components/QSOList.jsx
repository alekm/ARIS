import React, { useState, useEffect } from 'react';
import { Database, Trash2 } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || '';

const QSOList = () => {
    const [qsos, setQsos] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [selectedQSO, setSelectedQSO] = useState(null);

    useEffect(() => {
        const fetchQsos = async () => {
            try {
                // setLoading(true); // Don't block UI on refresh
                const res = await fetch(`${API_BASE}/api/qsos?limit=50`);
                if (!res.ok) {
                    throw new Error(`HTTP ${res.status}`);
                }
                const data = await res.json();
                setQsos(data);
                setError(null);
            } catch (e) {
                console.error('Error fetching QSOs:', e);
                setError(e.message);
            } finally {
                setLoading(false);
            }
        };
        fetchQsos();
        const interval = setInterval(fetchQsos, 5000); // Refresh every 5 seconds
        return () => clearInterval(interval);
    }, []);

    const formatTime = (datetime) => {
        if (!datetime) return 'N/A';
        try {
            const date = new Date(datetime);
            return date.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
        } catch (e) {
            return datetime.split('T')[1]?.split('.')[0] || datetime;
        }
    };

    const handleDelete = async (e, sessionId) => {
        e.stopPropagation();
        if (!window.confirm("Delete this record permanently?")) return;

        try {
            const res = await fetch(`${API_BASE}/api/qsos/${sessionId}`, { method: 'DELETE' });
            if (res.ok) {
                setQsos(prev => prev.filter(q => q.session_id !== sessionId));
                if (selectedQSO && selectedQSO.session_id === sessionId) {
                    setSelectedQSO(null);
                }
            } else {
                console.error("Delete failed");
                alert("Failed to delete record");
            }
        } catch (err) {
            console.error("Delete error:", err);
            alert("Error deleting record");
        }
    };

    return (
        <div className="panel" style={{ height: '100%', display: 'flex', flexDirection: 'column', position: 'relative' }}>
            <div className="panel-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Database size={16} />
                    <span>INTERCEPTED_COMMUNICATIONS_LOG</span>
                </div>
                <div style={{ fontSize: '0.8em', color: '#666' }}>
                    RECORDS: {qsos.length}
                </div>
            </div>

            <div style={{ padding: '15px', overflowY: 'auto', flex: 1 }}>
                {loading && qsos.length === 0 && (
                    <div style={{ textAlign: 'center', padding: '40px', color: '#666' }}>
                        LOADING...
                    </div>
                )}
                {error && (
                    <div style={{ textAlign: 'center', padding: '40px', color: '#f44' }}>
                        ERROR: {error}
                    </div>
                )}
                {!loading && qsos.length === 0 && !error && (
                    <div style={{ textAlign: 'center', padding: '40px', color: '#666' }}>
                        NO RECORDS FOUND
                    </div>
                )}
                {qsos.length > 0 && (
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9em' }}>
                        <thead>
                            <tr style={{ textAlign: 'left', color: '#666', borderBottom: '1px solid #333' }}>
                                <th style={{ padding: '8px' }}>TIME</th>
                                <th style={{ padding: '8px' }}>FREQ (kHz)</th>
                                <th style={{ padding: '8px' }}>MODE</th>
                                <th style={{ padding: '8px' }}>CALLSIGNS</th>
                                <th style={{ padding: '8px' }}>SUMMARY</th>
                                <th style={{ padding: '8px', width: '40px' }}></th>
                            </tr>
                        </thead>
                        <tbody>
                            {qsos.map((q, i) => (
                                <tr
                                    key={q.session_id || i}
                                    onClick={async () => {
                                        setSelectedQSO(q); // Show immediate with summary only
                                        try {
                                            const res = await fetch(`${API_BASE}/api/qsos/${q.session_id}`);
                                            if (res.ok) {
                                                const detailed = await res.json();
                                                setSelectedQSO(detailed);
                                            }
                                        } catch (e) {
                                            console.error("Failed to fetch details", e);
                                        }
                                    }}
                                    style={{
                                        borderBottom: '1px solid #222',
                                        cursor: 'pointer',
                                    }}
                                    onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#111'}
                                    onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                                >
                                    <td style={{ padding: '8px', color: '#aaa' }}>
                                        {formatTime(q.start_datetime)}
                                    </td>
                                    <td style={{ padding: '8px', color: 'var(--color-primary)' }}>
                                        {(q.frequency_hz / 1000).toFixed(1)}
                                    </td>
                                    <td style={{ padding: '8px' }}>{q.mode || 'N/A'}</td>
                                    <td style={{ padding: '8px', color: '#fff' }}>
                                        {q.callsigns && q.callsigns.length > 0 ? q.callsigns.join(', ') : 'N/A'}
                                    </td>
                                    <td style={{ padding: '8px', color: '#888', maxWidth: '300px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                        {q.summary || 'No summary'}
                                    </td>
                                    <td style={{ padding: '8px', textAlign: 'center' }}>
                                        <button
                                            onClick={(e) => handleDelete(e, q.session_id)}
                                            style={{
                                                background: 'none',
                                                border: 'none',
                                                color: '#666',
                                                cursor: 'pointer',
                                                padding: '4px'
                                            }}
                                            onMouseEnter={(e) => e.target.style.color = '#f44'}
                                            onMouseLeave={(e) => e.target.style.color = '#666'}
                                            title="Delete Log"
                                        >
                                            <Trash2 size={14} />
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </div>

            {/* Detail Modal */}
            {selectedQSO && (
                <div style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    backgroundColor: 'rgba(0,0,0,0.85)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    zIndex: 100,
                    backdropFilter: 'blur(2px)'
                }} onClick={() => setSelectedQSO(null)}>
                    <div style={{
                        backgroundColor: '#050505',
                        border: '1px solid var(--color-primary)',
                        width: '90%',
                        maxWidth: '800px',
                        maxHeight: '80%',
                        display: 'flex',
                        flexDirection: 'column',
                        boxShadow: '0 0 20px rgba(0, 255, 65, 0.2)'
                    }} onClick={e => e.stopPropagation()}>
                        <div className="panel-header">
                            <div>TRANSMISSION_DETAILS</div>
                            <button
                                onClick={() => setSelectedQSO(null)}
                                style={{
                                    background: 'none',
                                    border: 'none',
                                    color: '#666',
                                    cursor: 'pointer',
                                    fontSize: '1.2em'
                                }}
                            >×</button>
                        </div>
                        <div style={{ padding: '20px', overflowY: 'auto' }}>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '15px', marginBottom: '20px' }}>
                                <div>
                                    <div style={{ fontSize: '0.7em', color: '#666' }}>FREQUENCY</div>
                                    <div style={{ color: 'var(--color-primary)', fontSize: '1.2em' }}>
                                        {(selectedQSO.frequency_hz / 1000).toFixed(1)} kHz
                                    </div>
                                </div>
                                <div>
                                    <div style={{ fontSize: '0.7em', color: '#666' }}>MODE</div>
                                    <div style={{ fontSize: '1.2em' }}>{selectedQSO.mode}</div>
                                </div>
                                <div>
                                    <div style={{ fontSize: '0.7em', color: '#666' }}>TIME</div>
                                    <div>{formatTime(selectedQSO.start_datetime)}</div>
                                </div>
                                <div>
                                    <div style={{ fontSize: '0.7em', color: '#666' }}>CALLSIGNS</div>
                                    <div style={{ color: '#fff' }}>
                                        {selectedQSO.callsigns && selectedQSO.callsigns.length > 0
                                            ? selectedQSO.callsigns.join(', ')
                                            : 'None'}
                                    </div>
                                </div>
                            </div>

                            <div style={{ marginBottom: '20px' }}>
                                <div style={{ fontSize: '0.7em', color: '#666', marginBottom: '5px' }}>SUMMARY</div>
                                <div style={{
                                    backgroundColor: '#111',
                                    padding: '10px',
                                    border: '1px solid #333',
                                    lineHeight: '1.5',
                                    color: '#ddd'
                                }}>
                                    {selectedQSO.summary || 'No summary available.'}
                                </div>
                            </div>

                            <div>
                                <div style={{ fontSize: '0.7em', color: '#666', marginBottom: '5px' }}>TRANSCRIPT_LOG</div>
                                <div style={{
                                    backgroundColor: '#0a0a0a',
                                    border: '1px solid #333',
                                    height: '300px',
                                    overflowY: 'auto',
                                    padding: '10px'
                                }}>
                                    {!selectedQSO.transcripts ? (
                                        <div style={{ color: '#666', textAlign: 'center', padding: '20px' }}>
                                            LOADING_TRANSCRIPTS...
                                        </div>
                                    ) : selectedQSO.transcripts.length === 0 ? (
                                        <div style={{ color: '#666', textAlign: 'center', padding: '20px' }}>
                                            NO_TRANSCRIPTS_FOUND
                                        </div>
                                    ) : (
                                        selectedQSO.transcripts.map((t, i) => (
                                            <div key={i} style={{ marginBottom: '10px', borderBottom: '1px solid #222', paddingBottom: '5px' }}>
                                                <div style={{ display: 'flex', justifyContent: 'space-between', color: '#555', fontSize: '0.8em', marginBottom: '2px' }}>
                                                    <span>{t.datetime.split('T')[1].split('.')[0]} UTC</span>
                                                    <span>CONF: {(t.confidence * 100).toFixed(0)}%</span>
                                                </div>
                                                <div style={{ color: '#aaa' }}>{t.text}</div>
                                            </div>
                                        ))
                                    )}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default QSOList;

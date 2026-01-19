import React, { useState, useEffect } from 'react';
import { Database } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || `http://${window.location.hostname}:8000`;

const QSOList = () => {
    const [qsos, setQsos] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchQsos = async () => {
            try {
                setLoading(true);
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

    return (
        <div className="panel" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
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
                            </tr>
                        </thead>
                        <tbody>
                            {qsos.map((q, i) => (
                                <tr key={q.session_id || i} style={{ borderBottom: '1px solid #222' }}>
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
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </div>
        </div>
    );
};

export default QSOList;

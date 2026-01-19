import React, { useState, useEffect } from 'react';
import { Database, Search } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || `http://${window.location.hostname}:8000`;

const QSOList = () => {
    const [qsos, setQsos] = useState([]);

    useEffect(() => {
        const fetchQsos = async () => {
            try {
                const res = await fetch(`${API_BASE}/api/qsos?limit=50`);
                const data = await res.json();
                setQsos(data);
            } catch (e) {
                console.error(e);
            }
        };
        fetchQsos();
    }, []);

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
                            <tr key={i} style={{ borderBottom: '1px solid #222' }}>
                                <td style={{ padding: '8px', color: '#aaa' }}>
                                    {q.time_start.split('T')[1].split('.')[0]}
                                </td>
                                <td style={{ padding: '8px', color: 'var(--color-primary)' }}>
                                    {(q.frequency_hz / 1000).toFixed(1)}
                                </td>
                                <td style={{ padding: '8px' }}>{q.mode}</td>
                                <td style={{ padding: '8px', color: '#fff' }}>
                                    {q.callsigns.join(', ')}
                                </td>
                                <td style={{ padding: '8px', color: '#888', maxWidth: '300px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                    {q.summary}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default QSOList;

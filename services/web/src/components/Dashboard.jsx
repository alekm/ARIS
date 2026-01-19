import React, { useState, useEffect } from 'react';
import SlotCard from './SlotCard';
import TranscriptFeed from './TranscriptFeed';

const API_BASE = import.meta.env.VITE_API_URL || `http://${window.location.hostname}:8000`;

const Dashboard = () => {
    const [slots, setSlots] = useState([
        { id: 1, status: 'offline', activeConfig: null },
        { id: 2, status: 'offline', activeConfig: null },
        { id: 3, status: 'offline', activeConfig: null },
        { id: 4, status: 'offline', activeConfig: null }
    ]);
    const [transcripts, setTranscripts] = useState([]);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const transRes = await fetch(`${API_BASE}/api/transcripts?limit=50`);
                const transData = await transRes.json();
                setTranscripts(transData);

                const slotsRes = await fetch(`${API_BASE}/api/slots`);
                const activeSlots = await slotsRes.json();

                setSlots(prevSlots => prevSlots.map(slot => {
                    const active = activeSlots.find(s => String(s.id) === String(slot.id));
                    if (active) {
                        return {
                            ...slot,
                            status: active.status,
                            activeConfig: {
                                frequency_hz: active.frequency_hz,
                                mode: active.mode,
                                host: active.host,
                                port: active.port,
                                mode: active.mode // Demod mode
                            }
                        };
                    }
                    return slot;
                }));
            } catch (e) {
                console.error("Dashboard Poll error:", e);
            }
        };

        const interval = setInterval(fetchData, 2000);
        fetchData(); // Initial
        return () => clearInterval(interval);
    }, []);

    const handleStartSlot = async (id, config) => {
        try {
            await fetch(`${API_BASE}/api/slots/${id}/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });
            setSlots(prev => prev.map(s => s.id === id ? { ...s, status: 'online' } : s));
        } catch (e) {
            alert("Failed to start slot: " + e.message);
        }
    };

    const handleStopSlot = async (id) => {
        try {
            await fetch(`${API_BASE}/api/slots/${id}/stop`, {
                method: 'POST'
            });
            setSlots(prev => prev.map(s => s.id === id ? { ...s, status: 'offline' } : s));
        } catch (e) {
            alert("Failed to stop slot: " + e.message);
        }
    };

    return (
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(400px, 1fr) 1.5fr', gap: '20px', flex: 1, minHeight: 0 }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '15px', overflowY: 'auto' }}>
                {slots.map(slot => (
                    <SlotCard
                        key={slot.id}
                        slot={slot}
                        onStart={handleStartSlot}
                        onStop={handleStopSlot}
                    />
                ))}
            </div>

            <div style={{ height: '100%' }}>
                <TranscriptFeed transcripts={transcripts} />
            </div>
        </div>
    );
};

export default Dashboard;

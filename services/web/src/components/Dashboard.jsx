import React, { useState, useEffect } from 'react';
import SlotCard from './SlotCard';
import TranscriptFeed from './TranscriptFeed';
import { useWebSocket } from '../contexts/WebSocketContext';

const API_BASE = import.meta.env.VITE_API_URL || '';

const Dashboard = () => {
    const [slots, setSlots] = useState([
        { id: 1, status: 'offline', activeConfig: null },
        { id: 2, status: 'offline', activeConfig: null },
        { id: 3, status: 'offline', activeConfig: null },
        { id: 4, status: 'offline', activeConfig: null }
    ]);
    const [transcripts, setTranscripts] = useState([]);
    const { slotsData, lastMessage } = useWebSocket();

    // Initial Transcript Fetch (keep this for history)
    useEffect(() => {
        const fetchHistory = async () => {
            try {
                const res = await fetch(`${API_BASE}/api/transcripts?limit=50`);
                if (res.ok) {
                    const data = await res.json();
                    setTranscripts(data);
                }
            } catch (e) {
                console.error("Failed to fetch transcript history");
            }
        };
        fetchHistory();
    }, []);

    // Handle Real-time Slot Updates via WebSocket
    useEffect(() => {
        if (slotsData && slotsData.length > 0) {
            setSlots(prevSlots => prevSlots.map(slot => {
                const active = slotsData.find(s => String(s.id) === String(slot.id));
                if (active) {
                    return {
                        ...slot,
                        status: active.status,
                        activeConfig: {
                            frequency_hz: active.frequency_hz,
                            mode: active.mode,
                            host: active.host,
                            port: active.port
                        }
                    };
                }
                return slot;
            }));
        }
    }, [slotsData]);

    // Handle Real-time Transcripts
    useEffect(() => {
        if (lastMessage && lastMessage.type === 'TRANSCRIPT') {
            setTranscripts(prev => {
                // Deduplicate by ID if possible, or just prepend
                const newT = lastMessage.data;
                // Optional: Check if already exists (for slow networks/reconnects)
                if (prev.some(t => t.id === newT.id)) return prev;
                return [newT, ...prev].slice(0, 50);
            });
        }
    }, [lastMessage]);

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

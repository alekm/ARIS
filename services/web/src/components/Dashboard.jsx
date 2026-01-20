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
    const { slotsData, lastMessage, isConnected, connectionError } = useWebSocket();

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

    // Handle transcript ID updates via WebSocket (after persistence)
    useEffect(() => {
        if (lastMessage && lastMessage.type === 'TRANSCRIPT_ID_UPDATE') {
            const idUpdate = lastMessage.data;
            setTranscripts(prev => {
                // Find transcript matching timestamp + text and update its ID
                return prev.map(t => {
                    if (!t.id && 
                        Math.abs(t.timestamp - idUpdate.timestamp) < 0.1 && 
                        t.text === idUpdate.text) {
                        return { ...t, id: idUpdate.id };
                    }
                    return t;
                });
            });
        }
    }, [lastMessage]);

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
            const newT = lastMessage.data;
            console.log("Received transcript via WebSocket:", newT);
            setTranscripts(prev => {
                // Validate transcript data before adding
                if (!newT || !newT.text || !newT.timestamp) {
                    console.warn("Invalid transcript data received:", newT);
                    return prev;
                }
                
                // Deduplicate by timestamp + text (since we don't have ID from stream)
                // Check if we already have this transcript (same timestamp and text)
                const exists = prev.some(t => 
                    t.timestamp === newT.timestamp && 
                    t.text === newT.text
                );
                if (exists) {
                    console.log("Transcript already exists, skipping");
                    return prev;
                }
                console.log("Adding new transcript to feed");
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
            {connectionError && (
                <div style={{
                    gridColumn: '1 / -1',
                    padding: '10px',
                    background: '#ff4444',
                    color: '#fff',
                    borderRadius: '4px',
                    marginBottom: '10px'
                }}>
                    ⚠️ {connectionError}
                </div>
            )}
            {!isConnected && !connectionError && (
                <div style={{
                    gridColumn: '1 / -1',
                    padding: '10px',
                    background: '#ffaa00',
                    color: '#000',
                    borderRadius: '4px',
                    marginBottom: '10px'
                }}>
                    🔄 Connecting to WebSocket...
                </div>
            )}
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
                <TranscriptFeed 
                    transcripts={transcripts} 
                    onDelete={(transcriptId) => {
                        // Remove deleted transcript from local state
                        setTranscripts(prev => prev.filter(t => t.id !== transcriptId));
                    }}
                />
            </div>
        </div>
    );
};

export default Dashboard;

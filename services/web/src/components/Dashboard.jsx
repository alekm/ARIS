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
    const { slotsData, subscribe, isConnected, connectionError } = useWebSocket();

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

    // Sync slots with live SLOT_UPDATE data from WebSocket
    useEffect(() => {
        if (!slotsData || slotsData.length === 0) return;

        setSlots(prevSlots => {
            const prevMap = new Map(prevSlots.map(s => [s.id, s]));

            return slotsData.map(serverSlot => {
                const prev = prevMap.get(serverSlot.id) || {};
                const prevConfig = prev.activeConfig || {};

                const activeConfig = {
                    host: serverSlot.host ?? prevConfig.host ?? '127.0.0.1',
                    port: serverSlot.port ?? prevConfig.port ?? 8073,
                    frequency_hz: serverSlot.frequency_hz ?? prevConfig.frequency_hz ?? 7200000,
                    mode: serverSlot.mode ?? prevConfig.mode ?? 'USB',
                    source_type: serverSlot.source_type ?? prevConfig.source_type ?? 'kiwi'
                };

                return {
                    id: serverSlot.id,
                    status: serverSlot.status || prev.status || 'offline',
                    activeConfig
                };
            });
        });
    }, [slotsData]);

    // Subscribe to WebSocket updates
    useEffect(() => {
        // Handle transcript ID updates
        const unsubIdUpdate = subscribe('TRANSCRIPT_ID_UPDATE', (msg) => {
            const idUpdate = msg.data;
            setTranscripts(prev => {
                return prev.map(t => {
                    if (!t.id &&
                        Math.abs(t.timestamp - idUpdate.timestamp) < 0.1 &&
                        t.text === idUpdate.text) {
                        return { ...t, id: idUpdate.id };
                    }
                    return t;
                });
            });
        });

        // Handle Real-time Transcripts
        const unsubTranscript = subscribe('TRANSCRIPT', (msg) => {
            const newT = msg.data;
            console.log("Received transcript via WebSocket:", newT);
            setTranscripts(prev => {
                // Validate transcript data before adding
                if (!newT || !newT.text || !newT.timestamp) {
                    console.warn("Invalid transcript data received:", newT);
                    return prev;
                }

                // Deduplicate by timestamp + text (since we don't have ID from stream)
                const exists = prev.some(t =>
                    t.timestamp === newT.timestamp &&
                    t.text === newT.text
                );
                if (exists) {
                    // console.log("Transcript already exists, skipping");
                    return prev;
                }
                console.log("Adding new transcript to feed");
                return [newT, ...prev].slice(0, 50);
            });
        });

        return () => {
            if (unsubIdUpdate) unsubIdUpdate();
            if (unsubTranscript) unsubTranscript();
        };
    }, [subscribe]);

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

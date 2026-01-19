import React, { useState } from 'react';
import { Activity, Power, Settings } from 'lucide-react';

const SlotCard = ({ slot, onStart, onStop }) => {
    const [config, setConfig] = useState({
        host: slot.activeConfig?.host || '127.0.0.1',
        port: slot.activeConfig?.port || 8073,
        frequency_hz: slot.activeConfig?.frequency_hz || 7200000,
        demod_mode: slot.activeConfig?.mode || 'USB',
        mode: 'kiwi'
    });

    // Use ref to track previous server config to avoid overwriting user edits on every poll
    const prevServerConfig = React.useRef(null);

    // Update local state ONLY when slot prop content actually changes
    React.useEffect(() => {
        if (slot.activeConfig) {
            const configStr = JSON.stringify(slot.activeConfig);
            if (configStr !== prevServerConfig.current) {
                setConfig(prev => ({
                    ...prev,
                    host: slot.activeConfig.host || prev.host,
                    port: slot.activeConfig.port || prev.port,
                    frequency_hz: slot.activeConfig.frequency_hz || prev.frequency_hz,
                    demod_mode: slot.activeConfig.mode || prev.demod_mode
                }));
                prevServerConfig.current = configStr;
            }
        }
    }, [slot.activeConfig]);

    const isOnline = slot.status === 'online';

    const handleStart = () => {
        onStart(slot.id, config);
    };

    return (
        <div className="panel" style={{ position: 'relative' }}>
            <div className="panel-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Activity size={16} />
                    <span>RX_SLOT_{slot.id}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '0.8em', color: isOnline ? 'var(--color-primary)' : '#555' }}>
                        {isOnline ? 'LINK_ACTIVE' : 'OFFLINE'}
                    </span>
                    <div style={{
                        width: '6px', height: '6px',
                        background: isOnline ? 'var(--color-primary)' : '#333',
                        boxShadow: isOnline ? '0 0 5px var(--color-primary)' : 'none'
                    }} />
                </div>
            </div>

            <div style={{ padding: '15px' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '10px', marginBottom: '10px' }}>
                    <div>
                        <label className="grid-label">Target Host</label>
                        <input
                            className="input"
                            value={config.host}
                            onChange={e => setConfig({ ...config, host: e.target.value })}
                            disabled={isOnline}
                        />
                    </div>
                    <div>
                        <label className="grid-label">Port</label>
                        <input
                            className="input"
                            value={config.port}
                            onChange={e => setConfig({ ...config, port: parseInt(e.target.value) })}
                            disabled={isOnline}
                        />
                    </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '10px', marginBottom: '15px' }}>
                    <div>
                        <label className="grid-label">Frequency (kHz)</label>
                        <input
                            className="input"
                            value={config.frequency_hz / 1000}
                            onChange={e => setConfig({ ...config, frequency_hz: Math.round(parseFloat(e.target.value) * 1000) })}
                            disabled={isOnline}
                            type="number"
                            step="0.001"
                            style={{ color: isOnline ? 'var(--color-primary)' : 'inherit', fontWeight: 'bold' }}
                        />
                    </div>
                    <div>
                        <label className="grid-label">Mode</label>
                        <select
                            className="input"
                            value={config.demod_mode}
                            onChange={e => setConfig({ ...config, demod_mode: e.target.value })}
                            disabled={isOnline}
                        >
                            <option value="USB">USB</option>
                            <option value="LSB">LSB</option>
                            <option value="AM">AM</option>
                            <option value="CW">CW</option>
                        </select>
                    </div>
                </div>

                {/* Action Button */}
                {isOnline ? (
                    <button
                        className="btn btn-danger"
                        style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px' }}
                        onClick={() => onStop(slot.id)}
                    >
                        <Power size={14} /> TERMINATE LINK
                    </button>
                ) : (
                    <button
                        className="btn btn-primary"
                        style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px' }}
                        onClick={handleStart}
                    >
                        <Activity size={14} /> INITIALIZE LINK
                    </button>
                )}
            </div>
        </div>
    );
};

export default SlotCard;

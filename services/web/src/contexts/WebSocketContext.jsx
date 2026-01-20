import React, { createContext, useContext, useEffect, useState, useRef } from 'react';

const WebSocketContext = createContext({
    isConnected: false,
    slotsData: [],
    connectionError: null,
    subscribe: () => { }
});

export const WebSocketProvider = ({ children }) => {
    const [isConnected, setIsConnected] = useState(false);
    const [slotsData, setSlotsData] = useState([]);
    const [connectionError, setConnectionError] = useState(null);
    const socketRef = useRef(null);
    const reconnectTimeoutRef = useRef(null);
    const reconnectAttemptsRef = useRef(0);
    const isConnectingRef = useRef(false);

    // Subscribers: { [messageType]: [callback1, callback2, ...] }
    const listenersRef = useRef({});

    const subscribe = (type, callback) => {
        if (!listenersRef.current[type]) {
            listenersRef.current[type] = [];
        }
        listenersRef.current[type].push(callback);

        // Return unsubscribe function
        return () => {
            if (listenersRef.current[type]) {
                listenersRef.current[type] = listenersRef.current[type].filter(cb => cb !== callback);
            }
        };
    };

    // Connect logic
    useEffect(() => {
        const connect = () => {
            // Prevent multiple simultaneous connection attempts
            if (isConnectingRef.current || (socketRef.current && socketRef.current.readyState === WebSocket.CONNECTING)) {
                return;
            }

            // Clean up existing connection
            if (socketRef.current) {
                socketRef.current.onopen = null;
                socketRef.current.onclose = null;
                socketRef.current.onerror = null;
                socketRef.current.onmessage = null;
                if (socketRef.current.readyState === WebSocket.OPEN || socketRef.current.readyState === WebSocket.CONNECTING) {
                    socketRef.current.close();
                }
            }

            isConnectingRef.current = true;
            setConnectionError(null);

            // Determine API Base
            const apiBase = import.meta.env.VITE_API_URL || '';

            let wsUrl = '';
            if (apiBase.startsWith('http')) {
                // Absolute URL - convert http/https to ws/wss
                wsUrl = apiBase.replace(/^http/, 'ws') + '/ws';
            } else if (apiBase.startsWith('ws://') || apiBase.startsWith('wss://')) {
                // Already a WebSocket URL
                wsUrl = apiBase + '/ws';
            } else {
                // Relative path - use Vite proxy (works in dev and prod with nginx)
                // Just use /ws which Vite will proxy to the API service
                wsUrl = '/ws';
            }

            console.log("Connecting to WS:", wsUrl);

            try {
                const ws = new WebSocket(wsUrl);

                ws.onopen = () => {
                    console.log("WS Connected");
                    setIsConnected(true);
                    reconnectAttemptsRef.current = 0;
                    isConnectingRef.current = false;
                    setConnectionError(null);
                };

                ws.onclose = (evt) => {
                    console.log("WS Closed:", evt.code, evt.reason);
                    setIsConnected(false);
                    isConnectingRef.current = false;
                    socketRef.current = null;

                    // Don't reconnect on clean close (1000) or going away during page load (1001 on first attempt)
                    // 1001 = Going Away (server closed connection, often during initial connection)
                    // 1006 = Abnormal Closure (connection lost)
                    if (evt.code === 1000) {
                        // Clean close, don't reconnect
                        return;
                    }

                    // Reconnect for other close codes
                    if (reconnectAttemptsRef.current < 10) {
                        const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 30000); // Exponential backoff, max 30s
                        reconnectAttemptsRef.current++;
                        console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current})...`);
                        reconnectTimeoutRef.current = setTimeout(connect, delay);
                    } else if (reconnectAttemptsRef.current >= 10) {
                        setConnectionError("Max reconnection attempts reached. Please refresh the page.");
                    }
                };

                ws.onerror = (err) => {
                    console.error("WS Error:", err);
                    setConnectionError("WebSocket connection error");
                    isConnectingRef.current = false;
                };

                ws.onmessage = (event) => {
                    try {
                        const msg = JSON.parse(event.data);
                        // console.log("WS Message received:", msg.type, msg.data ? Object.keys(msg.data) : 'no data');

                        // Handle internal state updates
                        if (msg.type === 'SLOT_UPDATE') {
                            setSlotsData(msg.data || []);
                        }

                        // Dispatch to listeners
                        const listeners = listenersRef.current[msg.type];
                        if (listeners) {
                            listeners.forEach(cb => {
                                try {
                                    cb(msg);
                                } catch (err) {
                                    console.error("Error in WS listener:", err);
                                }
                            });
                        }
                    } catch (e) {
                        console.warn("Failed to parse WS message:", e);
                    }
                };

                socketRef.current = ws;
            } catch (err) {
                console.error("Failed to create WebSocket:", err);
                setConnectionError("Failed to create WebSocket connection");
                isConnectingRef.current = false;
                // Retry after delay
                reconnectTimeoutRef.current = setTimeout(connect, 3000);
            }
        };

        // Small delay to ensure Vite proxy is ready on initial load
        const initialDelay = setTimeout(() => {
            connect();
        }, 500);

        return () => {
            // Cleanup on unmount
            clearTimeout(initialDelay);
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current);
            }
            if (socketRef.current) {
                socketRef.current.onopen = null;
                socketRef.current.onclose = null;
                socketRef.current.onerror = null;
                socketRef.current.onmessage = null;
                if (socketRef.current.readyState === WebSocket.OPEN || socketRef.current.readyState === WebSocket.CONNECTING) {
                    socketRef.current.close(1000, "Component unmounting");
                }
                socketRef.current = null;
            }
            isConnectingRef.current = false;
        };
    }, []); // Empty deps - only run once on mount

    return (
        <WebSocketContext.Provider value={{ isConnected, slotsData, connectionError, subscribe }}>
            {children}
        </WebSocketContext.Provider>
    );
};

export const useWebSocket = () => useContext(WebSocketContext);

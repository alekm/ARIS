import React, { createContext, useContext, useEffect, useState, useRef } from 'react';

const WebSocketContext = createContext({
    isConnected: false,
    lastMessage: null,
    slotsData: []
});

export const WebSocketProvider = ({ children }) => {
    const [isConnected, setIsConnected] = useState(false);
    const [lastMessage, setLastMessage] = useState(null);
    const [slotsData, setSlotsData] = useState([]);
    const socketRef = useRef(null);

    // Connect logic
    useEffect(() => {
        let reconnectTimeout;

        const connect = () => {
            // Determine API Base
            // Use relative path to leverage Vite Proxy (dev) or Nginx (prod)
            const apiBase = import.meta.env.VITE_API_URL || '';

            let wsUrl = '';
            if (apiBase.startsWith('http')) {
                wsUrl = apiBase.replace(/^http/, 'ws') + '/ws';
            } else {
                // If apiBase is relative (e.g. "", "/api"), assume same origin (proxy)
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const host = window.location.host; // e.g. localhost:3000
                wsUrl = `${protocol}//${host}${apiBase}/ws`;
            }

            console.log("Connecting to WS:", wsUrl);

            const ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                console.log("WS Connected");
                setIsConnected(true);
            };

            ws.onclose = () => {
                console.log("WS Closed, reconnecting...");
                setIsConnected(false);
                reconnectTimeout = setTimeout(connect, 3000);
            };

            ws.onerror = (err) => {
                console.error("WS Error", err);
                ws.close();
            };

            ws.onmessage = (event) => {
                try {
                    const msg = JSON.parse(event.data);
                    setLastMessage(msg);

                    if (msg.type === 'SLOT_UPDATE') {
                        setSlotsData(msg.data);
                    }
                } catch (e) {
                    // ignore non-json
                }
            };

            socketRef.current = ws;
        };

        connect();

        return () => {
            if (socketRef.current) socketRef.current.close();
            clearTimeout(reconnectTimeout);
        };
    }, []);

    return (
        <WebSocketContext.Provider value={{ isConnected, lastMessage, slotsData }}>
            {children}
        </WebSocketContext.Provider>
    );
};

export const useWebSocket = () => useContext(WebSocketContext);

import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import QSOList from './components/QSOList';
import { Terminal, Database } from 'lucide-react';
import { useWebSocket } from './contexts/WebSocketContext';
import './App.css';

const API_BASE = import.meta.env.VITE_API_URL || '';

const Layout = ({ children }) => {
  const [stats, setStats] = useState(null);
  const location = useLocation();
  const { isConnected, lastMessage } = useWebSocket();

  // Receive stats via WebSocket instead of polling
  useEffect(() => {
    if (lastMessage && lastMessage.type === 'STATS') {
      setStats(lastMessage.data);
    }
  }, [lastMessage]);

  return (
    <div style={{
      maxWidth: '1600px',
      margin: '0 auto',
      padding: '20px',
      height: '100vh',
      display: 'flex',
      flexDirection: 'column',
      boxSizing: 'border-box'
    }}>
      {/* System Header */}
      <div style={{
        marginBottom: '20px',
        padding: '0 0 15px 0',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        borderBottom: '1px solid #333'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Terminal size={24} color="var(--color-primary)" />
            <div>
              <h1 style={{ margin: 0, fontSize: '1.2em', color: 'var(--color-text)', letterSpacing: '1px' }}>
                ARIS<span style={{ color: 'var(--color-primary)' }}>_COMMAND</span>
              </h1>
            </div>
          </div>

          {/* Navigation Links */}
          <nav style={{ display: 'flex', gap: '5px' }}>
            <Link to="/" style={{ textDecoration: 'none' }}>
              <button className={`btn ${location.pathname === '/' ? 'btn-primary' : ''}`}>
                DASHBOARD
              </button>
            </Link>
            <Link to="/qsos" style={{ textDecoration: 'none' }}>
              <button className={`btn ${location.pathname === '/qsos' ? 'btn-primary' : ''}`}>
                QSO_LOGS
              </button>
            </Link>
          </nav>
        </div>

        {stats && (
          <div style={{ display: 'flex', gap: '20px', fontSize: '0.8em', fontFamily: 'monospace', color: '#666' }}>
            <Link to="/" style={{ color: 'inherit', textDecoration: 'none', cursor: 'pointer' }}>
              <span>RECS: <span style={{ color: '#fff' }}>{stats.transcripts_count}</span></span>
            </Link>
            <Link to="/qsos" style={{ color: 'inherit', textDecoration: 'none', cursor: 'pointer' }}>
              <span>QSOS: <span style={{ color: '#fff' }}>{stats.qsos_count}</span></span>
            </Link>
            <span style={{ color: isConnected ? 'var(--color-primary)' : '#ffaa00' }}>
              {isConnected ? 'WS_CONNECTED' : 'WS_CONNECTING...'}
            </span>
          </div>
        )}
      </div>

      <div style={{ flex: 1, minHeight: 0 }}>
        {children}
      </div>
    </div>
  );
};

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/qsos" element={<QSOList />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;

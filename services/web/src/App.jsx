import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import QSOList from './components/QSOList';
import PerformanceDashboard from './components/PerformanceDashboard';
import { Terminal, Database } from 'lucide-react';
import { useWebSocket } from './contexts/WebSocketContext';
import { useAuth } from './contexts/AuthContext.jsx';
import './App.css';

const API_BASE = import.meta.env.VITE_API_URL || '';

const Layout = ({ children }) => {
  const { isConnected, subscribe } = useWebSocket();
  const [stats, setStats] = useState(null);
  const location = useLocation();
  const { logout } = useAuth();

  // Receive stats via WebSocket
  useEffect(() => {
    const unsubscribe = subscribe('STATS', (msg) => {
      if (msg.data) {
        setStats(msg.data);
      }
    });
    return unsubscribe;
  }, [subscribe]);

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
            <Link to="/perf" style={{ textDecoration: 'none' }}>
              <button className={`btn ${location.pathname === '/perf' ? 'btn-primary' : ''}`}>
                PERF_MONITOR
              </button>
            </Link>
            <button
              className="btn"
              style={{ marginLeft: '10px' }}
              onClick={logout}
            >
              LOGOUT
            </button>
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

const LoginPage = () => {
  const { login, error, loading } = useAuth();
  const [password, setPassword] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!password) return;
    await login(password);
  };

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#050505',
        color: '#fff',
        fontFamily: 'var(--font-mono, monospace)',
      }}
    >
      <div
        style={{
          border: '1px solid var(--color-primary)',
          padding: '30px',
          width: '100%',
          maxWidth: '400px',
          boxShadow: '0 0 20px rgba(0,255,65,0.15)',
          background: '#020202',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
          <Terminal size={24} color="var(--color-primary)" />
          <div>
            <div style={{ fontSize: '1.1em', letterSpacing: '1px' }}>
              ARIS<span style={{ color: 'var(--color-primary)' }}>_COMMAND</span>
            </div>
            <div style={{ fontSize: '0.7em', color: '#777' }}>AUTHORIZED ACCESS ONLY</div>
          </div>
        </div>

        <form onSubmit={handleSubmit}>
          <label
            style={{
              display: 'block',
              fontSize: '0.8em',
              marginBottom: '6px',
              color: '#aaa',
            }}
          >
            ADMIN PASSWORD
          </label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            style={{
              width: '100%',
              padding: '8px 10px',
              marginBottom: '12px',
              background: '#000',
              border: '1px solid #333',
              color: '#fff',
              fontFamily: 'inherit',
            }}
            autoFocus
          />
          {error && (
            <div style={{ color: '#ff5555', fontSize: '0.8em', marginBottom: '10px' }}>
              {error}
            </div>
          )}
          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading}
            style={{ width: '100%', marginTop: '5px' }}
          >
            {loading ? 'AUTHENTICATING...' : 'ENTER SYSTEM'}
          </button>
        </form>
      </div>
    </div>
  );
};

function App() {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return (
      <div
        style={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: '#050505',
          color: '#888',
          fontFamily: 'var(--font-mono, monospace)',
        }}
      >
        INITIALIZING_AUTH_SUBSYSTEM...
      </div>
    );
  }

  if (!isAuthenticated) {
    return <LoginPage />;
  }

  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/qsos" element={<QSOList />} />
          <Route path="/perf" element={<PerformanceDashboard />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;

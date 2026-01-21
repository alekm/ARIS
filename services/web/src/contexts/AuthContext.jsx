import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';

const API_BASE = import.meta.env.VITE_API_URL || '';

const AuthContext = createContext({
  isAuthenticated: false,
  loading: true,
  error: null,
  login: async () => {},
  logout: async () => {},
});

export const AuthProvider = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const refreshAuth = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/me`, {
        credentials: 'include',
      });
      if (res.ok) {
        const data = await res.json();
        setIsAuthenticated(!!data.authenticated);
      } else {
        setIsAuthenticated(false);
      }
    } catch {
      setIsAuthenticated(false);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshAuth();
  }, [refreshAuth]);

  const login = useCallback(async (password) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ password }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || 'Login failed');
      }
      setIsAuthenticated(true);
      setError(null);
    } catch (e) {
      setIsAuthenticated(false);
      setError(e.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  }, []);

  const logout = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      await fetch(`${API_BASE}/api/logout`, {
        method: 'POST',
        credentials: 'include',
      });
    } catch {
      // Ignore logout errors client-side
    } finally {
      setIsAuthenticated(false);
      setLoading(false);
    }
  }, []);

  return (
    <AuthContext.Provider
      value={{ isAuthenticated, loading, error, login, logout }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);


import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // Listen on all addresses
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://aris-api:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'http://aris-api:8000', // Use http://, Vite handles WS upgrade
        ws: true,
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path, // Don't rewrite the path
      }
    }
  }
})

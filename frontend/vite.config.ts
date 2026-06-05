import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  base: '/static/frontend/',
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:5001',
      '/livez': 'http://127.0.0.1:5001',
      '/readyz': 'http://127.0.0.1:5001',
    },
  },
})

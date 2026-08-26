import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  base: '/static/',
  publicDir: 'public',
  build: {
    outDir: '../web/static',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/session':   'http://localhost:8000',
      '/ws':        { target: 'ws://localhost:8000', ws: true },
      '/translate': 'http://localhost:8000',
      '/metrics':   'http://localhost:8000',
      '/health':    'http://localhost:8000',
    },
  },
});

import { defineConfig } from 'vite';

// base './' agar aplikasi bisa dijalankan dari root domain maupun subfolder (mis. GitHub Pages)
export default defineConfig({
  base: './',
  build: {
    target: 'es2020',
    outDir: 'dist',
  },
});

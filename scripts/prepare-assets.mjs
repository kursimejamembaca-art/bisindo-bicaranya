// Menyiapkan aset MediaPipe agar aplikasi bisa berjalan offline setelah dipasang:
// 1. menyalin file wasm dari paket npm ke public/wasm
// 2. mengunduh model hand_landmarker.task ke public/models (sekali saja)
import { cpSync, existsSync, mkdirSync, writeFileSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const wasmSrc = join(root, 'node_modules/@mediapipe/tasks-vision/wasm');
const wasmDst = join(root, 'public/wasm');
const modelUrl = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const modelDst = join(root, 'public/models/hand_landmarker.task');

if (existsSync(wasmSrc)) {
  mkdirSync(wasmDst, { recursive: true });
  cpSync(wasmSrc, wasmDst, { recursive: true });
  console.log('wasm disalin:', readdirSync(wasmDst).length, 'berkas');
} else {
  console.warn('Paket @mediapipe/tasks-vision belum terpasang. Jalankan npm install dulu.');
}

if (!existsSync(modelDst)) {
  mkdirSync(dirname(modelDst), { recursive: true });
  console.log('mengunduh model hand_landmarker...');
  const res = await fetch(modelUrl);
  if (!res.ok) throw new Error('Gagal mengunduh model: HTTP ' + res.status);
  writeFileSync(modelDst, Buffer.from(await res.arrayBuffer()));
  console.log('model tersimpan di public/models');
} else {
  console.log('model sudah ada, dilewati');
}

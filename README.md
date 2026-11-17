# BISINDO Bicara

Penerjemah bahasa isyarat BISINDO berbasis kamera dengan MediaPipe Hand Landmarker.

## Menjalankan

1. Pasang Node.js 18 atau lebih baru.
2. Jalankan `npm install`. Perintah ini juga menyalin file wasm dan mengunduh model
   `hand_landmarker.task` (sekitar 7 MB) ke `public/models`. Pengunduhan hanya dilakukan sekali.
3. Jalankan `npm run dev`, lalu buka alamat yang ditampilkan.

## Membangun untuk produksi

```
npm run build
```

Hasilnya ada di folder `dist/`. Folder ini bisa diunggah ke hosting statis apa pun.

## Catatan penting

- **Kamera hanya berjalan di HTTPS** (atau `localhost`). Jika dibuka di HP dari alamat IP biasa, kamera tidak akan aktif.
- Aplikasi memproses video di perangkat. Frame kamera tidak dikirim ke server mana pun.
- Data latih disimpan di penyimpanan browser (`localStorage`) di perangkat itu. Gunakan tombol Unduh di halaman Latih untuk membuat cadangan.
- Setelah dibuka sekali dengan koneksi, aplikasi dan model tersimpan oleh service worker sehingga bisa dipakai offline.

## Struktur

- `index.html` — tampilan
- `src/style.css` — gaya
- `src/main.js` — logika kamera, MediaPipe, pengenalan, dan penyimpanan
- `public/` — gambar, manifest, service worker, model, dan wasm
- `scripts/prepare-assets.mjs` — menyiapkan wasm dan model

## Menjalankan lewat GitHub Pages

1. Buat repositori baru di GitHub, misalnya `bisindo-bicara`.
2. Unggah seluruh isi folder ini ke repositori (lihat panduan `git` di bawah).
3. Buka **Settings → Pages**, lalu pada **Source** pilih **GitHub Actions**.
4. Setiap kali ada `push` ke branch `main`, workflow `.github/workflows/deploy.yml` akan membangun dan menerbitkan aplikasi.
5. Aplikasi bisa dibuka di `https://USERNAME.github.io/NAMA-REPO/`. Alamat HTTPS ini membuat kamera di HP bisa aktif.

Perintah `git` dari folder proyek:

```
git init
git add .
git commit -m "Versi awal BISINDO Bicara"
git branch -M main
git remote add origin https://github.com/USERNAME/NAMA-REPO.git
git push -u origin main
```

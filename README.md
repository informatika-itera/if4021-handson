<h1 align="center"><b>IF25-40305: Sistem Teknologi Multimedia</b></h1>
<h3 align="center">Multimedia: System & Technology (STM)</h3>

<p align="center">
  <b>Program Studi Teknik Informatika — Institut Teknologi Sumatera (ITERA)</b><br>
  Tahun Ajaran 2026/2027 · Semester Ganjil
</p>

<p align="center">
  <a href="https://mctm.web.id/course/if25-40305"><img src="https://img.shields.io/badge/Portal-mctm.web.id%2Fstm-0ea5e9?style=flat-square" alt="Portal Kuliah"></a>
  <a href="https://mctm.web.id/stm/rps"><img src="https://img.shields.io/badge/RPS-Silabus%20Lengkap-8b5cf6?style=flat-square" alt="Silabus"></a>
  <a href="https://mctm.web.id/stm/rules"><img src="https://img.shields.io/badge/Rules-Kontrak%20Kuliah-ec4899?style=flat-square" alt="Kontrak Kuliah"></a>
</p>

---

## 📌 Identitas Perkuliahan

| Informasi | Keterangan |
|---|---|
| **Mata Kuliah** | **Sistem Teknologi Multimedia** (*Multimedia: System & Technology*) |
| **Kode Mata Kuliah** | **IF25-40305** *(Kurikulum Sebelumnya: IF4021)* |
| **Dosen Penanggung Jawab** | **Martin C.T. Manullang, S.T., M.T., Ph.D.** |
| **Jadwal Kuliah** | **Kamis, 13.00 – 15.40 WIB** |

---

## 📖 Deskripsi Mata Kuliah

Mata kuliah **IF25-40305 Sistem Teknologi Multimedia (STM)** membahas konsep-konsep data multimedia mencakup data teks, audio, citra, dan video, serta integrasinya ke dalam sistem multimedia terpadu. Pendekatan perkuliahan berlandaskan **teori multimedia yang kokoh** dan **praktik langsung (*hands-on*)** menggunakan perangkat lunak standar industri.

Mahasiswa diajak mengeksplorasi representasi diskrit, kuantisasi, teorema Nyquist-Shannon, rantai konversi ADC-DAC, pengolahan sinyal dan filter audio, pemrosesan citra digital, spatial filtering, motion estimation pada video, pelacakan berbasis Vision AI (facial landmarks, pose estimation, hand tracking), jaringan multimedia & adaptive streaming (DASH/HLS), hingga teknologi mutakhir seperti *remote photoplethysmography* (rPPG).

---

## 💻 Lingkungan Komputasi & Toolchain

Praktikum dan tugas hands-on menggunakan ekosistem **Python 3.12**:

- **Python Version**: `Python 3.12` *(Disarankan menggunakan package manager [`uv`](https://docs.astral.sh/uv/) untuk isolasi environment yang cepat & ringan; alternatif `conda` / `venv` diperbolehkan)*.
- **Library Inti**:
  - **Audio Processing & MIR**: `librosa`, `soundfile`, `scipy`, `pyloudnorm`, `pedalboard`
  - **Citra & Kompresi**: `opencv-python` (`cv2`), `numpy`, `matplotlib`, `pillow`
  - **Vision AI, Landmark & Tracking**: `mediapipe`, `dlib`, `pyzbar`
  - **Tools & Multimedia Backend**: `jupyter` / `ipykernel`, `ffmpeg`

---

## 📂 Modul Hands-on Semester Berjalan (2026/2027 Ganjil)

Daftar modul notebook interaktif pada branch `main`:

| No | Modul | Topik Utama | Berkas Notebook |
|---|---|---|---|
| **01** | **Audio Resampling** | Downsampling, upsampling, aliasing & filter anti-aliasing | [`1_audio_resampling.ipynb`](notebook_handson/1_audio_resampling.ipynb) |
| **02** | **Visual Representation** | Waveform, spektrum FFT, spektrogram STFT, Mel-spec & RTA | [`2_audio_visual_representation.ipynb`](notebook_handson/2_audio_visual_representation.ipynb) |
| **03** | **Rekaman Kelas** | Analisis visual & restorasi rekaman kuliah di ruang kelas | [`3_audio_dikelas_visualization.ipynb`](notebook_handson/3_audio_dikelas_visualization.ipynb) |
| **04** | **Loudness & Normalization** | Peak dBFS, True Peak, RMS, Crest Factor, LUFS, Batch Normalization & Crossfade | [`4_audio_loudness_normalization.ipynb`](notebook_handson/4_audio_loudness_normalization.ipynb) |
| **05** | **Dynamic Processing** | Compressor, Brickwall Limiter, Noise Gate, Silence Trimming & Mastering Chain | [`5_audio_dynamic_processing.ipynb`](notebook_handson/5_audio_dynamic_processing.ipynb) |
| **06** | **Music Information Retrieval** | Pitch shift & time stretch, deteksi pitch (YIN/pYIN), onset/tempo/beat, chroma & kunci lagu (Krumhansl–Schmuckler), chord recognition, mini-Shazam | [`6_audio_mir.ipynb`](notebook_handson/6_audio_mir.ipynb) |
| **07** | **Struktur Data Citra & Warna** | ndarray (H, W, C), bit depth & overflow uint8, imread/imwrite, PNG vs JPEG, Grayscale, HSV, segmentasi warna & histogram | [`7_image_structure_color.ipynb`](notebook_handson/7_image_structure_color.ipynb) |

---

## 📚 Arsip Semester Sebelumnya (*Branch Archives*)

Materi hands-on dan modul semester-semester lampau tetap tersimpan rapi dan dapat diakses pada branch masing-masing:

| Semester | Branch GitHub | Sorotan Materi |
|---|---|---|
| **2025/2026 Genap** | [`2025_2`](https://github.com/informatika-itera/IF25-40305-handson/tree/2025_2) | Audio Processing, Image Processing, Video Processing, Landmark Detection (Week 1–14) |
| **2025/2026 Ganjil** | [`2025_1`](https://github.com/informatika-itera/IF25-40305-handson/tree/2025_1) | Audio Module, Image Processing, Face/Pose Tracking, Interactive Web Visualizations |
| **2024/2025 Genap** | [`2024_2`](https://github.com/informatika-itera/IF25-40305-handson/tree/2024_2) | Audio Manipulation, Image Compression, Video Processing, Facial Landmarks & rPPG |
| **2024/2025 Ganjil** | [`2024_1`](https://github.com/informatika-itera/IF25-40305-handson/tree/2024_1) | Audio FFT, Equalizer, Filter Audio, JPEG Compression, Respiratory Tracking |

> **Navigasi Lokal:**
> ```bash
> git switch <nama_branch>
> # Contoh untuk membuka materi 2025_2:
> git switch 2025_2
> # Untuk kembali ke materi aktif:
> git switch main
> ```

---

## 🚀 Panduan Unduh untuk Mahasiswa

Untuk mengunduh repositori materi semester aktif saat ini:

```bash
# Shallow clone (sangat cepat & hemat kuota internet):
git clone --depth 1 https://github.com/informatika-itera/IF25-40305-handson.git
cd IF25-40305-handson
```

---

## 🔗 Tautan Penting

- 🌐 **Portal Perkuliahan STM**: [https://mctm.web.id/course/if25-40305](https://mctm.web.id/course/if25-40305)
- 📋 **Silabus / RPS Lengkap**: [https://mctm.web.id/stm/rps](https://mctm.web.id/stm/rps)
- 📜 **Kontrak Kuliah Lengkap**: [https://mctm.web.id/stm/rules](https://mctm.web.id/stm/rules)

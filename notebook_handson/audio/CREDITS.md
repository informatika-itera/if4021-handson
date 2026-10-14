# Kredit & Lisensi Aset Audio

Seluruh berkas audio pada direktori ini digunakan untuk keperluan pendidikan pada mata kuliah
**IF25-40305 Sistem Teknologi Multimedia**, Institut Teknologi Sumatera (ITERA).

## `music_sample.wav`

| | |
|---|---|
| **Judul** | Vibe Ace |
| **Pencipta** | Kevin MacLeod |
| **Lisensi** | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| **Sumber** | https://freemusicarchive.org/music/Kevin_MacLeod/Jazz_Sampler/Vibe_Ace |
| **Modifikasi** | Potongan 10 detik (mulai detik ke-15,56), dikonversi ke mono 22.050 Hz 16-bit PCM, puncak dinormalisasi ke −3 dBFS |

Instrumen: vibraphone, piano, bass, dan drum kit — dipilih karena satu berkas ini sekaligus
memuat transien perkusi, deret harmonik, dan energi frekuensi rendah, sehingga cocok untuk
mendemonstrasikan keempat representasi visual audio.

## `speech_sample.wav`

| | |
|---|---|
| **Korpus** | LibriSpeech (SLR12) |
| **Lisensi** | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| **Sumber** | https://www.openslr.org/12/ |
| **Modifikasi** | Potongan 4 detik, mono 22.050 Hz 16-bit PCM |

## `audio_dikelas.m4a` / `audio_dikelas.wav`

| | |
|---|---|
| **Sumber** | Rekaman langsung di kelas IF25-40305, 24 September 2026 |
| **Asli** | `audio_dikelas.m4a` — AAC-LC, mono 48.000 Hz, 6,5 detik |
| **Modifikasi** | `audio_dikelas.wav` — dikonversi dengan `ffmpeg` ke mono 48.000 Hz 16-bit PCM, penguatan −8,8 dB agar puncak −1 dBFS (hasil decode AAC berpuncak ≈2,45 sehingga akan ter-*clipping* tanpa penurunan ini) |

## `drums_sample.wav`

| | |
|---|---|
| **Judul** | Choice - Drum-bass |
| **Pencipta** | Admiral Bob |
| **Lisensi** | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) |
| **Sumber** | http://ccmixter.org/files/admiralbob77/31574 (didistribusikan melalui `librosa` test data) |
| **Modifikasi** | Potongan 8 detik beat drum & bass, mono 22.050 Hz 16-bit PCM — ideal untuk demonstrasi transien perkusi, kompresi attack/release, dan crest factor tinggi |

## `classical_sample.wav`

| | |
|---|---|
| **Karya** | Hungarian Dance No. 5 in F-sharp minor |
| **Komponis** | Johannes Brahms |
| **Lisensi** | [Public Domain (Musopen)](https://musopen.org/) |
| **Sumber** | Musopen / Wikimedia Commons (didistribusikan melalui `librosa` test data) |
| **Modifikasi** | Potongan 10 detik dinamika orkestra gesek, mono 22.050 Hz 16-bit PCM — rentang dinamika lebar (*wide dynamic range*) |

## `podcast_sample.wav`

| | |
|---|---|
| **Korpus** | LibriSpeech ASR Corpus |
| **Pembaca** | Relawan LibriVox |
| **Lisensi** | [Public Domain](https://librivox.org/) |
| **Sumber** | https://www.openslr.org/12/ (didistribusikan melalui `librosa` test data) |
| **Modifikasi** | Potongan 9 detik dialog/narasi vokal, mono 22.050 Hz 16-bit PCM — ideal untuk normalisasi dialog, noise gating, dan silence trimming |

## `trumpet_sample.wav`

| | |
|---|---|
| **Judul** | solo trumpet 06 (*Jazz Trumpet Loops Pack in F, 90 bpm*) |
| **Pencipta** | Mihai Sorohan |
| **Lisensi** | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| **Sumber** | https://freesound.org/s/77711/ (didistribusikan melalui `librosa` test data) |
| **Modifikasi** | Berkas utuh 5,3 detik, mono 22.050 Hz 16-bit PCM, puncak dinormalisasi ke −3 dBFS — melodi monofonik untuk deteksi pitch |

## `nutcracker_sample.wav`

| | |
|---|---|
| **Judul** | Dance of the Sugar Plum Fairy |
| **Pencipta** | P.I. Tchaikovsky (komponis), Kevin MacLeod (aransemen) |
| **Lisensi** | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| **Sumber** | https://freemusicarchive.org/music/Kevin_MacLeod/Classical_Sampler/Dance_of_the_Sugar_Plum_Fairy (didistribusikan melalui `librosa` test data) |
| **Modifikasi** | 30 detik pertama, mono 22.050 Hz 16-bit PCM, puncak dinormalisasi ke −3 dBFS — deteksi kunci, chord, beat, dan fingerprinting |

## `brahms_sample.wav`

| | |
|---|---|
| **Karya** | Hungarian Dance No. 5 (aransemen string orchestra) |
| **Komponis / Pemain** | Johannes Brahms / US Army Strings |
| **Lisensi** | [Public Domain Mark 1.0](https://creativecommons.org/publicdomain/mark/1.0/) |
| **Sumber** | https://musopen.org/music/43805-hungarian-dance-no-5-in-f-sharp-minor-woo-1-string-orch (didistribusikan melalui `librosa` test data) |
| **Modifikasi** | 30 detik pertama, mono 22.050 Hz 16-bit PCM, puncak dinormalisasi ke −3 dBFS — studi kasus deteksi kunci |

## `snare_accelerate.wav`

| | |
|---|---|
| **Judul** | Accelerating snare drum |
| **Pencipta** | Brian McFee; sampel snare oleh sakebeats (https://freesound.org/people/sakebeats/sounds/176345/) |
| **Lisensi** | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| **Sumber** | `librosa` test data (`librosa.ex('snare')`) |
| **Modifikasi** | 20 detik pertama, mono 22.050 Hz 16-bit PCM, puncak dinormalisasi ke −3 dBFS — tempo yang terus meningkat untuk demonstrasi tempogram |

---

Lisensi CC BY 4.0 mewajibkan pencantuman atribusi. Berkas ini adalah pemenuhan kewajiban
tersebut — mohon dipertahankan apabila repositori disalin atau didistribusikan ulang.

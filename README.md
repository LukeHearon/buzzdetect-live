# buzzdetect live

buzzdetect's `model_general_v3` running in a browser tab, on a SeeNote-style
spectrogram. Open an audio file, see the spectrogram and the model's activations
on a shared time axis, play it back, scrub around. Or point it at your
microphone and watch both update live.

Nothing is uploaded. The file is decoded and analysed in the tab.

Live at [lukehearon.com/buzzdetect-live](https://lukehearon.com/buzzdetect-live).

```
npm install
npm run dev      # http://localhost:5180
npm test         # parity against buzzdetect's own numbers
```

## Does it agree with buzzdetect?

Yes, to well inside the precision buzzdetect itself reports.

On five minutes of soybean-field audio (313 frames), against the TensorFlow
SavedModels buzzdetect actually loads:

| measure | result |
| --- | --- |
| worst activation difference | 6.2e-3 |
| worst `ins_buzz` difference | 2.9e-3 (on a −2.17 … 1.47 range) |
| detections at `ins_buzz > −1.2` | 173 vs 173 — **zero flips** |
| cells differing at buzzdetect's 2-dp output | 9.7% (all by one unit in the last place) |

There are three checks, at increasing levels of realism:

- `tools/04_verify_onnx.py` — the ONNX graph against the SavedModels, fed
  identical patches. Isolates the export.
- `npm test` — the TypeScript log-mel front end against TensorFlow's features
  (max 2.3e-5), then the front end and model composed end to end (max 4.8e-3).
- `/parity.html` — the same comparison in the browser, through `fetch`,
  `decodeAudioData` and this machine's wasm build. Catches anything the browser
  introduces that node would not.

The shipped sample is 32-bit float 16 kHz mono — the exact array buzzdetect fed
its embedder — so the browser decodes it without resampling and the parity page
measures the model export rather than a decoder difference.

### What the difference is

Conv kernels are stored as float16 from the fifth block onward and expanded back
to float32 at load, which halves the download. Rounding error at a convolution
is amplified by every convolution after it, so the early blocks — where the
error has the most depth left to grow through, and where few of the parameters
live — stay float32. Compute is float32 throughout; only storage changes.

Alternatives measured on the same audio, all detection-identical except int8:

| storage | size | worst activation diff | detections |
| --- | --- | --- | --- |
| float32 | 12.8 MB | 7.4e-6 | 173 |
| **mixed (shipped)** | **6.5 MB** | **6.2e-3** | **173** |
| float16 throughout | 6.5 MB | 1.5e-2 | 173 |
| int8 per-channel | 3.3 MB | 3.5e-1 | 172 (3 flips) |

int8 is rejected: 0.155 of error on `ins_buzz` distorts the plotted curve and
the other twelve classes even where the thresholded call survives.

## What it costs to run

| | |
| --- | --- |
| model | 6.5 MB (gzip barely helps — weights are high entropy) |
| ONNX Runtime wasm | 13.5 MB raw, 3.4 MB gzip, 2.2 MB brotli |
| app code | 100 kB, 35 kB gzip |
| memory, 5 min of audio | ~19 MB of 16 kHz samples + ~8 MB of spectrogram, plus a transient ~58 MB display copy |

Both large assets are content-hashed and cached after the first visit.

Design choices that follow from caring about this:

- **Audio is decoded to mono, twice**: straight to 16 kHz for the model, and to a
  display rate that steps down with duration (48 kHz under 10 min, 24 kHz under
  30, 16 kHz beyond) for the spectrogram. A five-minute stereo 44.1 kHz file is
  ~19 MB as model input rather than the ~106 MB of holding it at native rate and
  stereo. The display copy is larger but transient — it is transferred to the
  worker and dropped the moment the spectrogram is built, and decoding twice
  avoids cascading two resamplers onto the path whose numbers have to match
  buzzdetect.
- **Playback uses an `<audio>` element**, not Web Audio, so a long recording
  costs nothing extra in memory and seeking is the browser's problem. You hear
  the file's full bandwidth; only the analysis is downsampled.
- **BatchNorm is folded into the convolutions** at export, removing 27 ops.
  Exact, not an approximation.
- **The mel filterbank is stored banded**, not as a 257x64 matrix that is 97%
  zeros — ~500 multiply-adds per frame instead of ~16,000.
- **The spectrogram is computed once into a byte buffer**, so panning and
  zooming never touch an FFT — they re-read the buffer, not the audio.
  `chooseHop` widens the hop for long files so the buffer stays under 64 MB
  whatever gets dropped in.
- **Scrolling is a blit.** Playback and live capture move the view every frame,
  and the image is already correct — only its position changes. The canvas is
  shifted onto itself and just the strip that scrolled into view is recomputed,
  turning a per-frame cost of width×height into a handful of columns. That is
  what lets the display run uncapped instead of the 30 Hz it needed when every
  frame was a full repaint.
- **The pixel loop caps how many columns it reads per screen pixel**, so zooming
  all the way out on a long file stays interactive.
- **All heavy work is in a worker**, and drawing is one rAF loop with dirty
  flags, so a burst of events costs one repaint rather than one per event.

## How the model was built

`tools/` runs in order. Steps 1, 2, 6 and 7 need the `buzzdetect` conda env and
must run from the buzzdetect repo root; steps 3–5 need only `tools/.venv`
(numpy, onnx, onnxruntime).

| | |
| --- | --- |
| `01_reference.py` | ground truth: runs real audio through buzzdetect's own path and saves waveform, log-mel, embeddings and activations. Asserts `yamnet.keras` matches the `yamnet_k2` SavedModel buzzdetect runs (5.7e-6) — the export is invalid otherwise. |
| `02_extract_weights.py` | pulls raw layer weights to `.npz` + a JSON spec. No arithmetic. |
| `03_build_onnx.py` | builds the single graph: YAMNet's conv stack with `model_general_v3`'s dense head fused on. Folds BatchNorm, resolves TF `SAME` padding to explicit pads, applies the storage precision. |
| `04_verify_onnx.py` | ONNX vs SavedModel on identical patches. |
| `05_detections.py` | detection-count agreement at a threshold — flips, not just totals. |
| `06_fixtures.py` | binary fixtures for the TypeScript tests. |
| `07_sample.py` | the shipped demo clip and its reference activations. |

Rebuild the model with:

```
tools/.venv/bin/python tools/03_build_onnx.py --precision mixed --mixed-from 4 --out buzzdetect_v3.onnx
cp tools/artifacts/buzzdetect_v3.onnx public/model/
```

## Layout

```
src/dsp/fft.ts            radix-2 FFT with a real-input specialisation
src/dsp/melspec.ts        YAMNet's log-mel front end; mirrors features.py exactly
src/dsp/spectrogram.ts    the display STFT, quantised to a byte per bin
src/model/session.ts      ONNX Runtime setup, class list, batching
src/workers/analysis.worker.ts   spectrogram, then streaming inference
src/audio/decode.ts       file -> 16 kHz mono + a blob URL for playback
src/audio/player.ts       <audio>-backed transport
src/live/liveSession.ts   microphone capture and scrolling spectrogram
src/ui/                   viewport, colormap, canvases, two-thumb range slider
src/parity.ts             the in-browser parity check
```

`src/dsp/melspec.ts` is the file to be careful with. Its constants and rounding
rules mirror `embedders/yamnet/features.py` and `params.py`; the padding
arithmetic reproduces TensorFlow's integer truncations and its float32 division
because those decide whether a file ends on N or N+1 frames. `test/melspec.test.ts`
pins the whole thing against TensorFlow's output.

## Activations are not probabilities

`model_general_v3`'s head is a single Dense layer with no softmax or sigmoid, so
its outputs are unbounded and do not sum to one. The app plots them in the
model's own units, as buzzdetect writes them to its `activation_*` columns. The
default threshold of −1.2 is in those units too.

## Deploying

Built and copied into the site by `buzzdetect-live.sh` in the website repo, as a
Quarto post-render step: `npm run build -- --base=/buzzdetect-live/` into
`_site/buzzdetect-live/`, plus a `.htaccess` carrying `COOP`/`COEP` and
`AddType application/wasm`. The build is skipped when nothing in this repo
changed. Test built output with `npm run preview` rather than `npm run dev` —
the model path resolves differently once bundled, and dev has hidden real
breakage there before.

## Limits

- **Formats** are whatever the browser decodes: wav, mp3, m4a/aac, ogg, flac.
  No wma, no mts — those need buzzdetect proper.
- **Long files.** Everything is held in memory; past ~20 minutes the app warns.
  There is no streaming analysis and no resume.
- **Threading** needs `SharedArrayBuffer`, so the page must be cross-origin
  isolated (`COOP`/`COEP`). Vite sets both headers in dev and preview. On a
  static host without them the runtime drops to one thread — slower, identical
  numbers.
- **Resampling.** Files that aren't already 16 kHz are resampled by the browser,
  where buzzdetect uses soxr via librosa. That difference is not measured by the
  parity page, which deliberately uses a 16 kHz sample to isolate the export.
  Quantifying it would need the same file decoded both ways.
- **Frame hop** is fixed at `framehop_prop = 1` (contiguous 0.96 s frames).
  The front end supports 0.5; nothing in the UI exposes it.
- No labelling, no subsetting, no project persistence. That's SeeNote's job.

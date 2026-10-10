# PixelLane

Windows image processing app with hand-written AVX2 SIMD filters and a scalar vs AVX2 benchmark.

## Features

- Grayscale conversion
- Sobel edge detection
- Gaussian blur with 1–3 passes
- Scalar vs AVX2 timing with a warmup run and median of multiple trials
- Automatic scalar/AVX2 output validation (`PASS - max diff 0`)
- PNG/JPG image saving
- Background worker thread so the UI stays responsive
- Runtime AVX2 detection with scalar fallback
- Per-monitor DPI awareness
- Dark neon-purple “Stealth Graphite” UI

## Requirements

- Windows 10 or Windows 11
- MinGW-w64 g++ and `windres`
- An AVX2-capable CPU is recommended; the application falls back to scalar processing when AVX2 is unavailable

## Build

Run the release build script from a MinGW-w64 shell or command prompt:

```bat
build_release.bat
```

The output is `PixelLane.exe`.

The release build uses static GCC and C++ standard libraries and does not enable
global `-mavx2`. AVX2 implementations are enabled selectively with
`target("avx2")`, while runtime CPU detection controls whether the AVX2 path is used.

## Project structure

- `gui_demo.cpp` — Win32 UI, worker threads, image loading/saving, and application entry point
- `image_processor.cpp` — scalar and AVX2 image-processing kernels; included by `gui_demo.cpp`
- `pixellane.rc` — Windows icon and version-information resource
- `app.ico` — multi-size application icon
- `build_release.bat` — MinGW-w64 release build script

## Screenshots

<!-- Add screenshots here. -->

# Raksha Kavach Standalone Offline Android APK

Self-contained Android `.apk` project powered by **Chaquopy** (`com.chaquo.python` v17.2.0) — the Python PII engine (`main.py`, `gui.py`, `src/`, `gui/`, `config/`) runs **inside the Android app itself** on `127.0.0.1`, with **no external server or Mac connection required**.

## Build the Standalone `.apk` in Android Studio

1. **Install Python 3.13** on your Mac (`https://www.python.org/downloads/macos/`) so Chaquopy can resolve the prebuilt Android 3.13 wheels (`PyYAML`, `regex`, `cryptography`, `phonenumbers`, `indic-transliteration`).
2. **Pull/clone the latest repo** (or re-download the latest ZIP after commit `17.2.0`) and open `android_app/` in **Android Studio**.
3. Wait for Gradle Sync (`Gradle 9.x` + `Chaquopy 17.2.0`) to finish downloading the Android Python wheels.
4. Click **Build → Build Bundle(s) / APK(s) → Build APK(s)**.
5. Install the generated `.apk` on your Android phone:
   `android_app/app/build/outputs/apk/debug/app-debug.apk`

## How It Runs Offline On-Device
- At app launch, `MainActivity.kt` starts CPython 3.13 in-process via Chaquopy (`AndroidPlatform`) and invokes `gui.start_background_server("127.0.0.1", 8573, filesDir.absolutePath)`.
- All outputs are written to the app's private writable storage (`filesDir/.gui_output`).
- `main.py` automatically runs single-process (`n_workers = 1`) and `src/detection.py` uses its pure-Python `difflib` fuzzy fallback when `rapidfuzz` is absent on Android.
- Use **Choose Files...** in Tab 1 to upload clinical files directly from your phone and **Download Sanitized (.zip)** to save results.

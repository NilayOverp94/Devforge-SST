# Raksha Kavach Standalone Offline Android APK

Self-contained Android `.apk` project powered by **Chaquopy** (`com.chaquo.python` v16.0.0) — the Python PII engine (`main.py`, `gui.py`, `src/`, `gui/`, `config/`) runs **inside the Android app itself** on `127.0.0.1`, with **no external server or Mac connection required**.

## Build the Standalone `.apk` in Android Studio

1. Install **Android Studio** (`https://developer.android.com/studio`) on your Mac.
2. Open Android Studio -> **Open** -> select this `android_app/` folder.
3. Make sure your Mac has Python 3.8–3.12 on `PATH` for Chaquopy's build step (or set `buildPython("/path/to/python3")` inside `chaquopy.defaultConfig` in `app/build.gradle.kts`) and wait for Gradle Sync to finish downloading the Android Python wheels (`PyYAML`, `phonenumbers`, `indic-transliteration`, `regex`, `cryptography`).
4. Click **Build → Build Bundle(s) / APK(s) → Build APK(s)**.
5. Install the generated `.apk` on any Android phone:
   `android_app/app/build/outputs/apk/debug/app-debug.apk`

## How It Runs Offline On-Device
- At app launch, `MainActivity.kt` starts CPython in-process via Chaquopy (`AndroidPlatform`) and invokes `gui.start_background_server("127.0.0.1", 8573, filesDir.absolutePath)`.
- All outputs are written to the app's private writable storage (`filesDir/.gui_output`).
- `main.py` automatically runs single-process (`n_workers = 1`) and `src/detection.py` uses its pure-Python `difflib` fuzzy fallback if `rapidfuzz` is absent on Android.
- Use **Choose Files...** in Tab 1 to upload clinical files directly from your phone and **Download Sanitized (.zip)** to save results.

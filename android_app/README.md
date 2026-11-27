# Raksha Kavach Android APK Wrapper

Lightweight Android `WebView` client for **Raksha Kavach**, built to open in **Android Studio** and compile directly into an `.apk`.

Because the full sanitization engine runs multi-process Python (`main.py` / `gui.py`) with C/Rust extensions (`rapidfuzz`, `cryptography`, `regex`, `jellyfish`) and handles 2 GB–4 GB datasets, the Python engine runs on your Mac/host while the Android APK renders the full 5-tab UI natively with file upload and ZIP download support.

## Build the `.apk` in Android Studio

1. Install **Android Studio** (`https://developer.android.com/studio`) on your Mac.
2. Open Android Studio -> **Open** -> select this `android_app/` folder.
3. Wait for Gradle Sync to download Android SDK/build tools.
4. Click **Build → Build Bundle(s) / APK(s) → Build APK(s)**.
5. Your APK is generated at:
   `android_app/app/build/outputs/apk/debug/app-debug.apk`

## Connect the APK to Your Local Server

Start the GUI server on your Mac:

```bash
cd /Users/nilayraj/Projecct/secret-wards-firewall
.venv/bin/python gui.py --host 0.0.0.0 --port 8573 --no-browser
```

Then choose one of three connection modes in the app's **Server** button:

- **Android Emulator:** tap **Emulator (10.0.2.2)** -> `http://10.0.2.2:8573/`
- **Physical Phone over USB:** run `adb reverse tcp:8573 tcp:8573`, then tap **USB (127.0.0.1)** -> `http://127.0.0.1:8573/`
- **Physical Phone on Same Wi-Fi:** enter your Mac's local IP (e.g. `http://192.168.1.10:8573/`) and tap **Connect**.

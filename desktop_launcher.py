#!/usr/bin/env python3
"""Frozen desktop app launcher for Raksha Kavach (.app/.dmg on macOS, .exe on Windows)."""
from __future__ import annotations

import multiprocessing
import sys
import threading
import webbrowser
from pathlib import Path

import gui
import main  # noqa: F401 — ensure PyInstaller bundles main.py and src/* workers


def _launch_desktop_app() -> int:
    data_dir = str((Path.home() / "RakshaKavachData").resolve()) if getattr(sys, "frozen", False) else None
    url = gui.start_background_server(host="127.0.0.1", port=8573, data_dir=data_dir)
    webbrowser.open(url)

    try:
        import tkinter as tk

        root = tk.Tk()
        root.title("Raksha Kavach")
        root.geometry("420x170")
        root.resizable(False, False)

        frame = tk.Frame(root, padx=18, pady=16)
        frame.pack(fill="both", expand=True)

        tk.Label(
            frame,
            text="Raksha Kavach Offline Console Running",
            font=("Helvetica", 13, "bold"),
        ).pack(anchor="w")

        tk.Label(
            frame,
            text=f"Local URL: {url}\n100% Offline • CPU-Only • Local Output in ~/RakshaKavachData",
            justify="left",
            fg="#475569",
            pady=8,
        ).pack(anchor="w")

        btns = tk.Frame(frame)
        btns.pack(fill="x", pady=(6, 0))

        tk.Button(btns, text="Open in Browser", command=lambda: webbrowser.open(url)).pack(side="left")
        tk.Button(btns, text="Quit Server", command=root.destroy).pack(side="right")

        root.mainloop()
    except Exception:
        threading.Event().wait()
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(_launch_desktop_app())

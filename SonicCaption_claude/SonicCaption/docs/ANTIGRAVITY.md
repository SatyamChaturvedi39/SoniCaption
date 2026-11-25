# Running in Antigravity

1. Open the `SonicCaption` folder (File → Open Folder).
2. Terminal → Run Task → **SonicCaption: setup** (first time only, a few minutes: it installs PyTorch).
3. Optional: copy `.env.example` to `.env` and add `GROQ_API_KEY`.
4. Run Task → **SonicCaption: run**, then open http://localhost:8000 in the browser (use the Antigravity browser or Chrome; allow microphone/camera).
5. Tests: Run Task → **SonicCaption: test**.

If the agent panel is used, point it at `AGENTS.md`. Debugging: launch config "SonicCaption API (debug)".

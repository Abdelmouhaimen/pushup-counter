# Pushup Counter

This is a small personal project built to see how far I could get with a browser-based pushup counter using pose detection.

Right now the app opens the webcam, tracks the arms in real time, shows a simple overlay, counts reps while recording is active, and lets you save the session as a `.webm` file. The current setup is aimed at a front-facing user and uses 2D pose landmarks for the elbow-angle logic because that has been more stable than the 3D landmarks for this use case.

The backend is intentionally minimal. Flask just serves the page and static assets, while the actual pose detection runs in the browser through MediaPipe Tasks Vision.

## Stack

- Python
- Flask
- MediaPipe Tasks Vision
- Plain HTML, CSS, and JavaScript

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000`.

You will need to allow webcam access in the browser. The page also pulls the MediaPipe runtime and model from a CDN, so an internet connection is required.

## Run with Docker

```bash
docker build -t pushup-counter .
docker run --rm -p 5000:5000 pushup-counter
```

Then open `http://localhost:5000`.

## Project layout

```text
.
|-- app.py
|-- Dockerfile
|-- requirements.txt
|-- templates/
|   `-- index.html
`-- static/
    |-- css/
    |   `-- style.css
    `-- js/
        `-- app.js
```

## Notes

- Rep counting starts only when recording is active.
- The overlay is meant for quick visual feedback, not perfect biomechanical measurement.
- The thresholds are simple and may need small adjustments depending on camera position and body proportions.

## Why I made it

Mostly because it seemed like a fun thing to build. I wanted a lightweight project that mixed a tiny Flask app with real-time computer vision in the browser, without turning it into a full product.

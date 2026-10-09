# CurrenSight AI 👁️💵

[![Python](https://img.shields.io/badge/Python-3.11%20|%203.12%20|%203.13%20|%203.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI%200.143-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch%20ResNet--18-EE4C2C.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV%20Headless-5C3EE8.svg)](https://opencv.org/)
[![Accessibility](https://img.shields.io/badge/Accessibility-WCAG%202.1%20AA-success.svg)](#accessibility-features)
[![Tests](https://img.shields.io/badge/Tests-20%20Passed-brightgreen.svg)](#tests)

> **Empowering Financial Independence for the Visually Impaired through Real-Time Computer Vision.**

CurrenSight AI is an accessibility-first Indian banknote denomination perception assistant. Powered by a hybrid PyTorch ResNet-18 and OpenCV vision pipeline, it provides sub-10ms real-time banknote identification via live webcam streaming, image uploads, natural voice announcements, and WCAG-compliant high-contrast controls.

---

## 🌟 About Us & Project Mission

### Why CurrenSight AI?
In India, while modern banknotes feature tactile bleeding lines and variable sizes, individuals with visual impairments, partial blindness, or age-related vision loss frequently encounter challenges in everyday transactions. Variations in lighting, soiled or folded notes, and rapid merchant exchanges create barriers to financial autonomy.

**CurrenSight AI was built to solve this challenge.** By bridging advanced deep learning with a universally accessible browser frontend, CurrenSight AI turns any smartphone, laptop, or webcam-equipped device into an instant, spoken banknote recognizer that works locally with zero cloud latency.

### Core Pillars
- **👁️ Accessibility-First Design:** Engineered from the ground up for screen-reader users, keyboard-only navigators, and low-vision individuals. Features an instant **High-Contrast Theme (Black & Yellow)**, custom focus states, and tactile keyboard shortcuts.
- **🗣️ Natural Voice Synthesis:** Integrates browser-native Web Speech API to immediately announce detected banknotes (e.g. *"Detected Five Hundred Rupees"*), with repeat-on-tap and background throttle prevention.
- **⚡ High-Performance Pipeline:** Optimized for low latency (~7.8 ms per inference, over **120 FPS** on CPU), ensuring fluid video scanning without frame buffering or thermal lag.
- **🛡️ Hybrid Vision Architecture:** Combines deep feature representations from a fine-tuned ResNet-18 with classic computer vision verification (HSV color histogram comparison, Laplacian texture sharpness, and Canny contour bounding-box localization).
- **🔒 Privacy by Design:** All frame processing occurs on the local FastAPI backend server; no user camera feeds or personal data are stored or shared.

### 👥 Meet the Team & Project Contributors
CurrenSight AI is our **Deep Learning & Computer Vision** project designed to deliver tangible accessibility impact:
- **Yash Sonawane** ([@yash12991](https://github.com/yash12991))
- **Bhavesh K. Sinha** ([@BhaveshKSinha](https://github.com/BhaveshKSinha))
- **Chetan** ([@Chetan0233](https://github.com/Chetan0233))
- **Rounak Ukey** ([@Rounak-U](https://github.com/Rounak-U))

**Mission:** Building open, ethical, and high-impact AI accessibility solutions for real-world independence. Contributions, issues, and feature requests are welcome on the [issues page](https://github.com/yash12991/RupeeScan-AI/issues).

---

## 📸 Key Features

| Feature | Description |
|---|---|
| **Live Webcam Stream** | Real-time WebSocket connection (`/ws/detect`) running at up to 120+ FPS |
| **Static Photo Upload** | Multi-part image upload REST endpoint (`POST /api/detect`) with client-side preview |
| **Voice Synthesis** | Spoken denomination announcements with debounced repeat handling |
| **High Contrast Mode** | Instant toggle between glassmorphic dark theme and high-contrast yellow-on-black |
| **Banknote Localization** | Real-time contour bounding-box tracking overlaid on the video feed |
| **Supported Notes** | **₹10, ₹20, ₹50, ₹100, ₹200, ₹500, and ₹2000** banknotes |
| **Accessibility Shortcuts** | <kbd>Alt + C</kbd> Contrast, <kbd>Alt + V</kbd> Voice, <kbd>Alt + A</kbd> About Us, <kbd>Space</kbd> Scan |

---

## 🏗️ System Architecture

![CurrenSight AI System Architecture](assets/architecture_diagram.png)

```mermaid
graph LR
    subgraph Input["1. Input Layer"]
        CAM["📷 Webcam / Mobile Camera"] --> CANVAS["Canvas Frame Grabbing"]
        UPLOAD["📁 Photo Upload (REST)"] --> ENCODE["Base64 JPEG Encoding"]
        CANVAS --> WS_CLIENT["WebSocket Client"]
    end

    subgraph Backend["2. Backend & Networking (FastAPI)"]
        WS_CLIENT --> WS_ENDPOINT["/ws/detect (WebSocket)"]
        ENCODE --> WS_ENDPOINT
        UPLOAD --> REST_ENDPOINT["/api/detect (POST)"]
        WS_ENDPOINT --> VALIDATE["Security & Pixel Validation (Pillow)"]
        REST_ENDPOINT --> VALIDATE
    end

    subgraph Vision["3. AI & Computer Vision Engine"]
        VALIDATE --> RESNET["PyTorch ResNet-18 (Denomination Classifier)"]
        VALIDATE --> OPENCV_COLOR["OpenCV HSV Color Correlation"]
        VALIDATE --> OPENCV_TEXTURE["Laplacian & Canny Texture Checks"]
        VALIDATE --> OPENCV_BBOX["Contour Bounding-Box Localization"]
        RESNET --> FUSION["Decision Fusion & Confidence Gating"]
        OPENCV_COLOR --> FUSION
        OPENCV_TEXTURE --> FUSION
        OPENCV_BBOX --> FUSION
    end

    subgraph Output["4. Accessible Output Layer"]
        FUSION --> JSON_OUT["JSON Result Payload"]
        JSON_OUT --> SPEECH["🗣️ Web Speech API (Spoken Audio)"]
        JSON_OUT --> BBOX_UI["🟩 Dynamic Banknote Bounding Box"]
        JSON_OUT --> CONTRAST["👁️ WCAG High-Contrast Display"]
    end

    style Input fill:#121420,stroke:#00f2fe,stroke-width:2px,color:#fff
    style Backend fill:#121420,stroke:#89f7fe,stroke-width:2px,color:#fff
    style Vision fill:#121420,stroke:#e040fb,stroke-width:2px,color:#fff
    style Output fill:#121420,stroke:#00e676,stroke-width:2px,color:#fff
```

---

## 🔍 What We Use (Technology Stack & Tooling)

| Component | Technology | Why We Use It |
|---|---|---|
| **Deep Learning** | **PyTorch & Torchvision (ResNet-18)** | Compact, battle-tested convolutional network fine-tuned on Indian banknote denominations; achieves sub-8ms CPU latency (~127 FPS). |
| **Computer Vision** | **OpenCV (opencv-python-headless)** | Fast image processing for color consistency (HSV histograms), print sharpness (Laplacian variance), and real-time banknote contour boundary detection. |
| **Backend Framework** | **FastAPI** | High-performance asynchronous Python web framework with native WebSocket support, threadpool offloading, and automated schema validation. |
| **Server Engine** | **Uvicorn (ASGI)** | Lightning-fast ASGI web server capable of handling persistent WebSocket frame streaming without memory leaks. |
| **Image Validation** | **Pillow (PIL)** | Safe preliminary image dimension and byte verification before decoding, preventing decompression bombs or malformed buffers. |
| **Voice Synthesis** | **Browser Web Speech API** | Client-side, zero-latency text-to-speech engine that converts detection verdicts into natural spoken Hindi/English voice guidance. |
| **Frontend Architecture** | **Vanilla HTML5, CSS3, & Modern JS** | Lightweight, responsive, zero-dependency browser client with zero build overhead. Runs instantaneously on mobile and desktop browsers. |
| **Accessibility Compliance** | **WCAG 2.1 AA Standards** | Specially calibrated high-contrast theme (Black & Yellow), screen-reader ARIA live regions, and tactile keyboard shortcuts (<kbd>Alt+C</kbd>, <kbd>Alt+V</kbd>, <kbd>Alt+A</kbd>, <kbd>Space</kbd>). |

---

## ⚙️ How It Works (End-to-End Pipeline)

### 1. Frame Capture & Streaming
The user opens CurrenSight AI in any browser. `navigator.mediaDevices.getUserMedia` activates the device camera (or user selects an image file). The browser client draws video frames onto an internal canvas, converts them into base64 JPEG buffers, and streams them over a bidirectional WebSocket (`/ws/detect`). A ping-pong control flag ensures the client only dispatches new frames once the previous response has arrived, preventing buffer bloat.

### 2. Validation & Preprocessing
FastAPI receives the frame and passes the raw buffer through `decode_image_bytes()`:
- Verifies the payload does not exceed **10 MB** (`CURRENSIGHT_MAX_IMAGE_BYTES`).
- Verifies decoded dimensions do not exceed **16 Megapixels** (`CURRENSIGHT_MAX_IMAGE_PIXELS`).
- Decodes the verified buffer into a standard OpenCV BGR NumPy array.

### 3. Stage 1: PyTorch ResNet-18 Denomination Classification
The image is converted to RGB, resized to `224x224`, normalized using standard ImageNet parameters, and passed through a custom fine-tuned **ResNet-18**:
- Evaluates output logits across all 7 Indian banknote classes: **₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000**.
- Applies softmax normalization to compute confidence scores.
- A confidence threshold (`> 0.55`) filters out blank backgrounds or non-currency objects.

### 4. Stage 2: OpenCV Visual Verification & Guardrails
To prevent false positives from plain paper or black-and-white drawings, OpenCV evaluates:
- **Color Consistency:** Compares the HSV color distribution of the query image against reference templates using histogram correlation (`cv2.compareHist`).
- **Texture & Edge Density:** Calculates Laplacian variance (`cv2.Laplacian`) and Canny edge density to ensure the frame contains high-frequency intaglio printing patterns characteristic of authentic banknotes rather than hand-drawn sketches.

### 5. Banknote Boundary Localization
The frame undergoes Gaussian blurring, Canny edge detection, and morphological dilation. Contours covering at least 4% of the image area with an aspect ratio between 1.2 and 3.6 are extracted. The coordinates are normalized into relative bounding box values `{"x", "y", "w", "h"}`.

### 6. Accessible Spoken & Visual Feedback
The JSON response is transmitted back to the browser client:
- The frontend dynamically aligns the visual bounding-box overlay around the banknote.
- The UI status text updates with high-contrast badge colors.
- The **Web Speech API** speaks the friendly denomination name (e.g., *"Detected Five Hundred Rupees"*). If the user double-taps anywhere on the screen or presses <kbd>Space</kbd>, the application immediately re-announces the last detected note.

---

## 🔬 In-Depth Computer Vision & Deep Learning Pipeline

After a video frame or uploaded photo arrives at the backend (`POST /api/detect` or `WS /ws/detect`), it passes through a multi-stage dual-track processing pipeline designed for speed, accuracy, and anti-spoofing resilience:

![CurrenSight AI In-Depth Computer Vision Pipeline](assets/cv_pipeline_in_depth.png)

```mermaid
flowchart TD
    IMG["📥 Client Image Received (Base64 / Multipart)"] --> DEC["🛡️ Safe Decompression & Validation (Pillow/OpenCV)"]
    DEC --> FORK{"Dual-Track Parallel Processing"}

    subgraph DeepLearning["Track A: Deep Learning Path (PyTorch ResNet-18)"]
        FORK --> RGB["Convert BGR ➔ RGB"]
        RGB --> RESIZE["Spatial Resize to 224 × 224 px"]
        RESIZE --> NORM["Tensor Normalization (ImageNet μ, σ)"]
        NORM --> RESNET["ResNet-18 Convolutional Forward Pass"]
        RESNET --> SOFTMAX["Softmax Activation ➔ 7 Class Probabilities"]
        SOFTMAX --> GATING{"Confidence > 0.55?"}
    end

    subgraph ClassicalCV["Track B: Classical Computer Vision Guardrails (OpenCV)"]
        FORK --> HSV["HSV Color Space Conversion"]
        HSV --> HIST["2D Histogram (32×32) Correlation Check"]
        FORK --> LAP["Laplacian Variance (Sharpness Check)"]
        FORK --> CANNY["Canny Edge Density (Intaglio Texture)"]
        HIST --> GUARD{"Color & Texture Valid?"}
        LAP --> GUARD
        CANNY --> GUARD
    end

    subgraph ContourLoc["Track C: Banknote Localization"]
        FORK --> BLUR["Gaussian Blur (5×5)"]
        BLUR --> EDGE["Canny Edge Detection (30, 150)"]
        EDGE --> DILATE["Morphological Dilation"]
        DILATE --> CONTOURS["Find External Contours"]
        CONTOURS --> BBOX["Filter Area (>4%) & Aspect Ratio (1.2 - 3.6)"]
    end

    GATING -- Yes --> FUSION["Decision Fusion Engine"]
    GUARD -- Pass --> FUSION
    BBOX --> FUSION
    GATING -- No --> UNKNOWN["Return Unrecognized / Align Note"]
    GUARD -- Fail --> UNKNOWN

    FUSION --> JSON_OUT["📤 Output JSON: Denomination, Confidence, Bounding Box"]

    style DeepLearning fill:#121420,stroke:#89f7fe,stroke-width:2px,color:#fff
    style ClassicalCV fill:#121420,stroke:#00e676,stroke-width:2px,color:#fff
    style ContourLoc fill:#121420,stroke:#e040fb,stroke-width:2px,color:#fff
    style FUSION fill:#1a1c2e,stroke:#00f2fe,stroke-width:2px,color:#fff
```

### Detailed Pipeline Stages

#### 1. Payload Ingestion & Defensive Decompression
- Incoming frames are checked against byte-size limits ($\le 10$ MB) and decoded dimensions ($\le 16$ Megapixels).
- Using Pillow's `Image.open().verify()`, corrupt bytes and decompression bomb attacks are rejected before OpenCV memory allocation.
- The validated buffer is converted into an 8-bit unsigned integer NumPy buffer and decoded via `cv2.imdecode()` to an $H \times W \times 3$ BGR matrix.

#### 2. Deep Learning Track (PyTorch ResNet-18)
- **Color Correction:** OpenCV's default BGR order is converted to RGB via `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.
- **Transformation Pipeline:**
  $$\text{Input Image} \xrightarrow{\text{Resize}(224 \times 224)} \xrightarrow{\text{ToTensor}} \xrightarrow{\text{Normalize}(\mu, \sigma)} \text{Tensor } [1, 3, 224, 224]$$
  where $\mu = [0.485, 0.456, 0.406]$ and $\sigma = [0.229, 0.224, 0.225]$.
- **Inference:** A fine-tuned **ResNet-18** executes on the PyTorch inference device (MPS/CUDA or CPU). Its skip connections ($F(x) + x$) preserve low-level currency watermarks and high-level structural features.
- **Classification:** The output logits pass through a softmax function to produce class posterior probabilities for the 7 denominations:
  $$P(\text{class}_i) = \frac{e^{z_i}}{\sum_{j=0}^{6} e^{z_j}}$$
- A confidence threshold ($\tau = 0.55$) prevents false positive triggers on plain backgrounds.

#### 3. Classical Computer Vision Guardrails Track
- **HSV 2D Histogram Correlation:**
  - Converts the image to Hue-Saturation-Value color space: `cv2.cvtColor(img, cv2.COLOR_BGR2HSV)`.
  - Calculates a 2D histogram over Hue $[0, 180]$ and Saturation $[0, 256]$ with $32 \times 32$ bins.
  - Compares the histogram against genuine banknote reference templates using the correlation metric:
    $$d(H_1, H_2) = \frac{\sum_I (H_1(I) - \bar{H}_1)(H_2(I) - \bar{H}_2)}{\sqrt{\sum_I (H_1(I) - \bar{H}_1)^2 \sum_I (H_2(I) - \bar{H}_2)^2}}$$
  - If the correlation score drops below the safe threshold ($0.535$), the frame is flagged for color mismatch (e.g., black-and-white printouts or wrong notes).
- **Texture & Edge Density Analysis:**
  - Evaluates print sharpness via the Laplacian variance: $\text{Var}(\nabla^2 I) \ge 1300$.
  - Evaluates high-frequency Canny edge density: $\ge 11.0\%$ of pixels. Real currency notes feature complex geometric security guilloches and intaglio printing that hand-drawn sketches or blank paper cannot mimic.

#### 4. Banknote Boundary Localization & Contour Analysis
- Converts frame to grayscale and applies a $5 \times 5$ Gaussian blur ($\sigma = 0$) to eliminate camera sensor noise.
- Applies Canny edge detection with hysteresis thresholds $(30, 150)$.
- Dilates the edges using a $5 \times 5$ rectangular structuring element to close disconnected perimeter segments.
- Extracts external contours using `cv2.findContours(cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`.
- **Geometric Filtering:**
  - Minimum area requirement: must cover $\ge 4\%$ of the total frame area.
  - Aspect ratio requirement: $1.2 \le \frac{\max(w, h)}{\min(w, h)} \le 3.6$, matching physical Indian currency proportions.
- Normalizes coordinates into relative units: $\{x, y, w, h\} \in [0.0, 1.0]$.

#### 5. Output Synthesis
- Combines the ResNet-18 classification, confidence rating, OpenCV guardrail validations, and normalized bounding box coordinates into a unified response JSON payload.
- Transmitted back to the client in under 10 milliseconds.

## ⚠️ Safety and Project Status

- **Denomination detection is the primary feature.** Authenticity classification is experimental, disabled by default, and must not be treated as proof that a banknote is genuine or counterfeit. Confirm questionable notes through an authorized bank or other official process.
- The YOLO scripts under `training/` are an offline research path for synthetic banknote localization. YOLO is not used by the deployed backend; the live application uses ResNet/classic-CV denomination classification and contour-based localization.
- Detailed model limitations and deployment policy are recorded in [MODEL_CARD.md](MODEL_CARD.md).

---

## 💻 Requirements

- **Operating System:** macOS, Linux, or Windows
- **Python:** 3.11 through 3.14 (fully verified on Python 3.12 and 3.14)
- A modern web browser with camera permissions (Chrome, Edge, Safari, Firefox)
- Pre-trained model weights under `backend/models/`

---

## 🚀 Quickstart & Installation

### 🍎 macOS / 🐧 Linux

```bash
# 1. Clone the repository
git clone https://github.com/yash12991/RupeeScan-AI.git
cd RupeeScan-AI

# 2. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r backend/requirements.txt

# 4. Verify system readiness with the built-in diagnostic doctor
python scripts/doctor.py

# 5. Launch the application
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### 🪟 Windows (PowerShell)

```powershell
# 1. Clone the repository
git clone https://github.com/yash12991/RupeeScan-AI.git
cd RupeeScan-AI

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt

# 4. Run diagnostic doctor
python scripts\doctor.py

# 5. Launch the application
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open your browser at **`http://127.0.0.1:8000`**. Camera APIs function automatically on `localhost`.

---

## ⚙️ Configuration

CurrenSight AI can be configured via environment variables:

| Variable | Purpose | Default |
|---|---|---|
| `CURRENSIGHT_MODEL_DIR` | Directory containing model weights (fallbacks: `DRISHTISCAN_*`, `RUPEESCAN_*`) | `backend/models` |
| `CURRENSIGHT_FRONTEND_DIR` | Static frontend directory (fallback: `RUPEESCAN_FRONTEND_DIR`) | `frontend` |
| `CURRENSIGHT_MAX_IMAGE_BYTES` | REST/WebSocket image payload limit | `10485760` (10 MB) |
| `CURRENSIGHT_MAX_IMAGE_PIXELS` | Maximum decoded image pixels | `16000000` (16 MP) |
| `CURRENSIGHT_ENABLE_EXPERIMENTAL_AUTHENTICITY` | Opt in to experimental authenticity inference | `false` |
| `CURRENSIGHT_DENOMINATION_DATASET` | Real denomination training dataset | `datasets/denomination` |
| `CURRENSIGHT_AUTHENTICITY_DATASET` | Independent real/counterfeit research dataset | `datasets/authenticity` |

---

## 🧪 Tests & Diagnostics

### Run Unit and Integration Tests
```bash
pip install -r requirements-dev.txt
pytest
```

The 20-test test suite verifies:
- API health and restrictive browser security headers (`X-Frame-Options`, `Content-Security-Policy`, etc.)
- Upload payload size and pixel dimension limits
- YOLO synthetic dataset split balance and labeling
- Diagnostic doctor contract and model artifact presence
- Frontend accessibility contracts (WCAG attributes, ARIA live-regions, unique element IDs)
- Shipped banknote smoke images inference verification

### Environment Doctor
```bash
python scripts/doctor.py
# Machine-readable output:
python scripts/doctor.py --json
```

### Inference Latency Benchmark
Measure CPU/GPU inference throughput on your system:
```bash
python scripts/benchmark_inference.py --iterations 30
```

---

## 🌐 API Reference

- `GET /api/live` — Lightweight liveness check for container and orchestration probes.
- `GET /api/health` — Comprehensive system diagnostic: model readiness, active engine, device, input limits, and supported classes.
- `POST /api/detect` — Single-image multipart upload endpoint returning denomination, confidence, and bounding box.
- `WS /ws/detect` — Real-time base64 WebSocket video frame processing stream.

---

## 🐳 Docker Deployment

CurrenSight AI includes a secure, non-root multi-stage Docker configuration:

```bash
# Build Docker image
docker build -t currensight:latest .

# Run container on port 8000
docker run --rm -p 8000:8000 currensight:latest
```

Access the application at `http://127.0.0.1:8000`.

---

## 📊 Dataset Evaluation & Training

### Real-World Evaluation
Evaluate performance against an independent directory of photographed notes (`real_photos/<denomination>/`):

```bash
python evaluation/evaluate_denomination.py path/to/real_photos --output evaluation/reports/real_photos.json
```

### PyTorch Model Training
```bash
# 1. Prepare deterministic 70/15/15 dataset manifests
python training/prepare_denomination_data.py

# 2. Dry run training validation
python training/train_pytorch.py --dry-run

# 3. Train ResNet-18 model
python training/train_pytorch.py --epochs 20 --batch-size 32 --patience 4
```

---

## 📄 License & Attribution

This project is open-source under the MIT License. Developed as a **Deep Learning & Computer Vision** project by **[Yash Sonawane](https://github.com/yash12991)**, **[Bhavesh K. Sinha](https://github.com/BhaveshKSinha)**, **[Chetan](https://github.com/Chetan0233)**, and **[Rounak Ukey](https://github.com/Rounak-U)**. If you use CurrenSight AI in your research, coursework, or project, please star the repository! ⭐

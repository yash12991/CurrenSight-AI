/* --- CurrenSight AI Application Controller --- */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const video = document.getElementById("webcam");
    const contrastBtn = document.getElementById("toggle-contrast");
    const speechBtn = document.getElementById("toggle-speech");
    const bboxOverlay = document.getElementById("bbox-overlay");
    const bboxLabel = bboxOverlay ? bboxOverlay.querySelector(".bbox-label") : null;
    const cameraFallback = document.getElementById("camera-fallback");
    const retryCameraBtn = document.getElementById("retry-camera");
    const connectionIndicator = document.getElementById("connection-indicator");
    const displayArea = document.getElementById("detection-display");
    const mOnIcon = speechBtn.querySelector(".m-on");
    const mOffIcon = speechBtn.querySelector(".m-off");
    const voiceBtnText = speechBtn.querySelector(".btn-text");

    // Static upload elements
    const imageUpload = document.getElementById("image-upload");
    const imagePreview = document.getElementById("image-preview");
    const resetCameraBtn = document.getElementById("reset-camera-btn");

    // About modal elements
    const aboutBtn = document.getElementById("open-about-modal");
    const aboutModal = document.getElementById("about-modal");
    const closeAboutBtn = document.getElementById("close-about-modal");
    const dismissAboutBtn = document.getElementById("dismiss-about-btn");

    // Audio Visualizer
    const audioVisualizer = document.getElementById("audio-visualizer");

    // Off-screen canvas for capturing/resizing video frames
    const captureCanvas = document.createElement("canvas");
    const captureCtx = captureCanvas.getContext("2d");

    // Application State Variables
    let webSocket = null;
    let cameraStream = null;
    let voiceEnabled = true;
    let isSocketConnecting = false;
    let frameIntervalId = null;
    let reconnectTimerId = null;
    let reconnectAttempts = 0;
    let isShuttingDown = false;
    let lastSpokenText = "";
    let lastSpokenTime = 0;
    let lastDetectedCurrency = "No Note";
    let isProcessingFrame = false; // Ping-pong flag to control WebSocket frame rate
    let candidateClass = null;
    let candidateCount = 0;
    let unknownFrameCount = 0;
    let lastRenderedKey = null;

    const MAX_CLIENT_UPLOAD_BYTES = 10 * 1024 * 1024;

    // Double/Triple Tap Gesture Variables
    let lastTapTime = 0;
    let tapCount = 0;
    let tapTimeout = null;

    // Initialize TTS Greeting
    speakVoice("CurrenSight is ready. Hold currency in front of camera.");

    // --- ACCESSIBILITY TTS FUNCTIONS ---

    function speakVoice(text, priority = false) {
        if (!voiceEnabled) return;

        const currentTime = Date.now();
        // Prevent spelling identical phrases unless 4 seconds have passed or priority is true
        if (text === lastSpokenText && (currentTime - lastSpokenTime < 4000) && !priority) {
            return;
        }

        // Cancel previous speaking to announce new detections instantly
        window.speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        utterance.lang = "en-IN"; // Use Indian English voice if available

        utterance.onstart = () => {
            lastSpokenText = text;
            lastSpokenTime = Date.now();
            if (audioVisualizer) audioVisualizer.classList.remove("hidden");
        };

        utterance.onend = () => {
            if (audioVisualizer) audioVisualizer.classList.add("hidden");
        };

        utterance.onerror = () => {
            if (audioVisualizer) audioVisualizer.classList.add("hidden");
        };

        window.speechSynthesis.speak(utterance);
    }

    function announceStatus(text) {
        // The detection card is replaced dynamically, so resolve the live region
        // each time instead of retaining a reference to a detached element.
        const statusText = document.getElementById("status-text");
        if (statusText) {
            statusText.textContent = text;
            statusText.setAttribute("aria-label", text);
        } else {
            // Successful-result cards replace status-text, while the parent remains
            // the persistent aria-live region.
            displayArea.setAttribute("aria-label", text);
        }
    }

    function clamp(value, minimum, maximum) {
        const number = Number(value);
        return Number.isFinite(number) ? Math.min(maximum, Math.max(minimum, number)) : minimum;
    }

    function escapeHtml(value) {
        return String(value ?? "").replace(/[&<>'"]/g, character => ({
            "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
        })[character]);
    }

    // --- WEBCAM STREAM HANDLING ---

    async function startCamera() {
        cameraFallback.classList.add("hidden");

        if (!navigator.mediaDevices?.getUserMedia) {
            announceStatus("Camera is not supported by this browser.");
            speakVoice("Camera access is not supported by this browser.");
            cameraFallback.classList.remove("hidden");
            return;
        }
        
        const constraints = {
            video: {
                facingMode: "environment", // Request back camera
                width: { ideal: 640 },
                height: { ideal: 480 }
            },
            audio: false
        };

        try {
            if (cameraStream) {
                cameraStream.getTracks().forEach(track => track.stop());
            }

            cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
            video.srcObject = cameraStream;
            
            // Set capture canvas size matching video resolution once loaded
            video.onloadedmetadata = () => {
                captureCanvas.width = Math.min(video.videoWidth || 640, 640);
                captureCanvas.height = Math.round(
                    captureCanvas.width * ((video.videoHeight || 480) / (video.videoWidth || 640))
                );
                announceStatus("Camera active. Align currency.");
                speakVoice("Camera loaded. Ready to scan.");
                
                // Start websocket and frame capture loop
                connectWebSocket();
            };
        } catch (error) {
            console.error("Camera access error:", error);
            announceStatus("Camera access denied.");
            speakVoice("Camera access failed. Please grant permission.");
            cameraFallback.classList.remove("hidden");
        }
    }

    // --- WEBSOCKET REAL-TIME FRAME STREAMING ---

    function connectWebSocket() {
        if (isShuttingDown || document.hidden) return;
        if (webSocket && (webSocket.readyState === WebSocket.OPEN || webSocket.readyState === WebSocket.CONNECTING)) {
            return;
        }

        if (reconnectTimerId) {
            clearTimeout(reconnectTimerId);
            reconnectTimerId = null;
        }

        isSocketConnecting = true;
        updateConnectionStatus("connecting");
        
        // Construct WebSocket URL matching backend location
        const loc = window.location;
        let wsUri = "";
        if (loc.protocol === "https:") {
            wsUri = "wss:";
        } else {
            wsUri = "ws:";
        }
        wsUri += "//" + loc.host + "/ws/detect";

        webSocket = new WebSocket(wsUri);

        webSocket.onopen = () => {
            isSocketConnecting = false;
            reconnectAttempts = 0;
            updateConnectionStatus("connected");
            announceStatus("Connected. Align banknote in viewfinder.");
            speakVoice("Connected to recognition engine.");
            
            // Reset frame processing lock
            isProcessingFrame = false;
            
            // Start the frame push interval
            startFrameStreaming();
        };

        webSocket.onmessage = (event) => {
            isProcessingFrame = false; // Unlock for next frame
            if (document.hidden) return;
            
            try {
                const data = JSON.parse(event.data);
                handleDetectionResult(data, "live");
            } catch (err) {
                console.error("Failed to parse prediction result:", err);
            }
        };

        webSocket.onclose = () => {
            isSocketConnecting = false;
            updateConnectionStatus("disconnected");
            stopFrameStreaming();
            
            if (isShuttingDown) return;

            // Exponential backoff avoids a tight reconnect loop while the server is down.
            const retryDelay = Math.min(30000, 1000 * (2 ** reconnectAttempts));
            reconnectAttempts += 1;
            reconnectTimerId = setTimeout(() => {
                reconnectTimerId = null;
                if (!isSocketConnecting && cameraStream && cameraStream.active && !document.hidden) {
                    announceStatus("Reconnecting to engine...");
                    connectWebSocket();
                }
            }, retryDelay);
        };

        webSocket.onerror = (error) => {
            console.error("WebSocket error:", error);
            webSocket.close();
        };
    }

    function updateConnectionStatus(status) {
        connectionIndicator.className = "badge";
        if (status === "connected") {
            connectionIndicator.textContent = "Live";
            connectionIndicator.classList.add("badge-success");
        } else if (status === "connecting") {
            connectionIndicator.textContent = "Connecting";
            connectionIndicator.classList.add("badge-error");
        } else {
            connectionIndicator.textContent = "Offline";
            connectionIndicator.classList.add("badge-error");
        }
        connectionIndicator.setAttribute("aria-label", `Recognition engine ${connectionIndicator.textContent}`);
    }

    function startFrameStreaming() {
        stopFrameStreaming();
        
        // Push frame every 200ms if websocket is free (ping-pong model)
        frameIntervalId = setInterval(() => {
            if (webSocket && webSocket.readyState === WebSocket.OPEN) {
                if (!isProcessingFrame) {
                    sendVideoFrame();
                }
            }
        }, 150); // ~6 FPS maximum client push, adapts automatically if server is slower
    }

    function stopFrameStreaming() {
        if (frameIntervalId) {
            clearInterval(frameIntervalId);
            frameIntervalId = null;
        }
    }

    function sendVideoFrame() {
        if (!video.videoWidth || !webSocket || webSocket.readyState !== WebSocket.OPEN) return;
        
        isProcessingFrame = true; // Lock frame push

        // Draw current camera frame onto hidden canvas
        captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
        
        // Compress canvas image to JPEG format (0.75 quality is optimal compression/detail balance)
        const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.75);
        
        // Send base64 frame content (removing the scheme prefix)
        const base64Data = dataUrl.substring(dataUrl.indexOf(",") + 1);
        try {
            webSocket.send(base64Data);
        } catch (error) {
            isProcessingFrame = false;
            console.error("Failed to send camera frame:", error);
        }
    }

    // --- DETECTION RESULT PROCESSING ---

    function handleDetectionResult(res, source = "live") {
        // If result is class_id -1 (unknown) or confidence too low
        if (res.class_id === -1) {
            unknownFrameCount += 1;
            if (source === "live" && unknownFrameCount < 3) return;
            candidateClass = null;
            candidateCount = 0;
            if (lastRenderedKey === "unknown" && source === "live") return;
            lastRenderedKey = "unknown";
            displayArea.innerHTML = `
                <div class="no-detection-msg">
                    <span class="pulse-dot"></span>
                    <p id="status-text">${escapeHtml(res.friendly_name || "Align banknote in frame")}</p>
                </div>
            `;
            announceStatus(res.friendly_name || "Align banknote in frame");
            if (bboxOverlay) bboxOverlay.classList.add("hidden");
            return;
        }

        unknownFrameCount = 0;
        if (source === "live") {
            if (candidateClass === res.class_name) {
                candidateCount += 1;
            } else {
                candidateClass = res.class_name;
                candidateCount = 1;
            }
            // Require two consecutive live frames before announcing a denomination.
            if (candidateCount < 2) return;
        }

        // Successfully detected banknote
        lastDetectedCurrency = res.friendly_name;
        const confidence = clamp(res.confidence, 0, 1);
        
        // Update bounding box tracker overlay
        if (res.bbox && bboxOverlay) {
            const bboxX = clamp(res.bbox.x, 0, 1);
            const bboxY = clamp(res.bbox.y, 0, 1);
            const bboxWidth = clamp(res.bbox.w, 0, 1 - bboxX);
            const bboxHeight = clamp(res.bbox.h, 0, 1 - bboxY);
            bboxOverlay.classList.remove("hidden");
            bboxOverlay.style.left = `${bboxX * 100}%`;
            bboxOverlay.style.top = `${bboxY * 100}%`;
            bboxOverlay.style.width = `${bboxWidth * 100}%`;
            bboxOverlay.style.height = `${bboxHeight * 100}%`;
            
            // Format label (e.g. "100 RUPEES")
            const denomLabel = res.class_name ? res.class_name.replace("Rs.", "").replace("_", " ").toUpperCase() : "BANKNOTE";
            if (bboxLabel) bboxLabel.textContent = denomLabel;
            
            // Adjust coloring based on authenticity
            if (res.authenticity_experimental && res.authentic === false) {
                bboxOverlay.classList.add("fake-border");
                bboxOverlay.classList.remove("real-border");
            } else if (res.authenticity_experimental && res.authentic === true) {
                bboxOverlay.classList.add("real-border");
                bboxOverlay.classList.remove("fake-border");
            } else {
                bboxOverlay.classList.remove("real-border", "fake-border");
            }
        } else if (bboxOverlay) {
            bboxOverlay.classList.add("hidden");
        }
        
        // Handle authenticity metadata if present
        let authClass = "";
        let authBadgeHtml = "";
        let voiceMessage = res.friendly_name;
        let prioritySpeak = false;

        if (res.authenticity_experimental && res.authentic !== null && res.authentic !== undefined) {
            if (res.authentic) {
                authClass = "real-card";
                authBadgeHtml = `<span class="auth-badge badge-real">LIKELY GENUINE · EXPERIMENTAL</span>`;
            } else {
                authClass = "fake-card";
                authBadgeHtml = `<span class="auth-badge badge-fake">POSSIBLE COUNTERFEIT · EXPERIMENTAL</span>`;
                voiceMessage = `Experimental warning. Possible counterfeit. ${res.friendly_name}`;
                prioritySpeak = true; // Interrupt normal voice for counterfeits
            }
        }

        // Update Dashboard Display
        const renderKey = `${res.class_name}|${res.authenticity_experimental ? res.authentic : "none"}`;
        const shouldRender = source === "upload" || renderKey !== lastRenderedKey;
        const denomThemeClass = `denom-${res.class_name || "unknown"}`;
        const confPercent = (confidence * 100).toFixed(0);
        const bboxInfo = res.bbox ? `BBox: ${(res.bbox.w * 100).toFixed(0)}% × ${(res.bbox.h * 100).toFixed(0)}%` : "Contour Centered";

        if (shouldRender) displayArea.innerHTML = `
            <div class="detection-result-card ${denomThemeClass} ${authClass}">
                <div class="result-header-row">
                    <span class="denom-symbol-badge">₹</span>
                    <div class="result-title-group">
                        <p class="result-denom">${escapeHtml(res.friendly_name)}</p>
                        <span class="denom-tag-pill">INDIAN BANKNOTE</span>
                    </div>
                </div>
                ${authBadgeHtml}
                <div class="result-conf-container">
                    <div class="conf-labels">
                        <span class="conf-text">AI Confidence</span>
                        <span class="conf-percentage">${confPercent}%</span>
                    </div>
                    <div class="result-conf-bar">
                        <div class="result-conf-fill" style="width: ${confidence * 100}%"></div>
                    </div>
                </div>
                <div class="detection-pills-grid">
                    <div class="det-pill">
                        <span class="det-pill-label">Engine</span>
                        <span class="det-pill-value">ResNet-18</span>
                    </div>
                    <div class="det-pill">
                        <span class="det-pill-label">Frame</span>
                        <span class="det-pill-value">${bboxInfo}</span>
                    </div>
                    <div class="det-pill">
                        <span class="det-pill-label">CV Guards</span>
                        <span class="det-pill-value">HSV &amp; Texture OK</span>
                    </div>
                </div>
                ${res.authenticity_confidence !== null && res.authenticity_confidence !== undefined ? 
                  `<p class="result-meta">Authenticity Confidence: ${(clamp(res.authenticity_confidence, 0, 1) * 100).toFixed(0)}%</p>` : ''}
                ${res.authenticity_experimental ?
                  `<p class="result-meta">Experimental estimate only. Do not rely on it for financial decisions.</p>` : ''}
            </div>
        `;
        lastRenderedKey = renderKey;
        
        if (shouldRender) announceStatus(`Detected ${res.friendly_name}`);
        
        // Speak results (handles debouncing inside speakVoice)
        if (shouldRender || source === "upload") speakVoice(voiceMessage, prioritySpeak);
        
        // Haptic feedback (Vibrate mobile device: triple vibrate for warning, single for normal)
        if ((shouldRender || source === "upload") && navigator.vibrate) {
            if (res.authenticity_experimental && res.authentic === false) {
                navigator.vibrate([100, 50, 100, 50, 200]);
            } else {
                navigator.vibrate(120);
            }
        }
    }

    // --- ACCESSIBILITY TOUCH GESTURE HANDLERS ---

    // Full screen gesture catcher
    document.body.addEventListener("click", (e) => {
        // Ignore clicks on buttons/controls
        if (e.target.closest("button") || e.target.closest("a") || e.target.closest("kbd")) {
            return;
        }

        const currentTime = Date.now();
        const tapGap = currentTime - lastTapTime;

        if (tapGap < 350) {
            tapCount++;
        } else {
            tapCount = 1;
        }

        lastTapTime = currentTime;

        if (tapTimeout) clearTimeout(tapTimeout);

        tapTimeout = setTimeout(() => {
            if (tapCount === 1) {
                // Single Tap: Announce current state
                if (lastDetectedCurrency && lastDetectedCurrency !== "No Note") {
                    speakVoice(`Last scan: ${lastDetectedCurrency}`, true);
                } else {
                    speakVoice("Scanning... Align note in frame.", true);
                }
            } else if (tapCount === 2) {
                // Double Tap: Repeat last detection with priority override
                if (lastDetectedCurrency) {
                    speakVoice(lastDetectedCurrency, true);
                } else {
                    speakVoice("No banknote detected yet.", true);
                }
            } else if (tapCount >= 3) {
                // Triple Tap: Toggle Contrast Theme
                toggleContrastTheme();
            }
            tapCount = 0;
        }, 360);
    });

    // --- EVENT CONTROLLER HANDLERS ---

    function toggleContrastTheme() {
        document.body.classList.toggle("high-contrast");
        const isActive = document.body.classList.contains("high-contrast");
        contrastBtn.classList.toggle("active", isActive);
        contrastBtn.setAttribute("aria-pressed", String(isActive));
        
        if (isActive) {
            speakVoice("High Contrast Theme Activated", true);
        } else {
            speakVoice("Normal Visual Theme Activated", true);
        }
    }

    function toggleSpeechEnabled() {
        voiceEnabled = !voiceEnabled;
        speechBtn.classList.toggle("active", voiceEnabled);
        speechBtn.setAttribute("aria-pressed", String(voiceEnabled));
        speechBtn.setAttribute("aria-label", voiceEnabled ? "Mute Voice Guidance" : "Enable Voice Guidance");
        
        if (voiceEnabled) {
            mOnIcon.classList.remove("hidden");
            mOffIcon.classList.add("hidden");
            voiceBtnText.textContent = "Voice On";
            speakVoice("Voice guidance unmuted", true);
        } else {
            // Cancel any speaking immediately
            window.speechSynthesis.cancel();
            if (audioVisualizer) audioVisualizer.classList.add("hidden");
            mOnIcon.classList.add("hidden");
            mOffIcon.classList.remove("hidden");
            voiceBtnText.textContent = "Voice Muted";
        }
    }

    // Keyboard Access Controls
    document.addEventListener("keydown", (e) => {
        // Toggle Contrast Theme (Alt+C)
        if (e.altKey && (e.key === "c" || e.key === "C")) {
            e.preventDefault();
            toggleContrastTheme();
        }
        // Toggle Voice Guidance (Alt+V)
        if (e.altKey && (e.key === "v" || e.key === "V")) {
            e.preventDefault();
            toggleSpeechEnabled();
        }
        // Toggle About Dialog (Alt+A)
        if (e.altKey && (e.key === "a" || e.key === "A")) {
            e.preventDefault();
            if (aboutModal && !aboutModal.classList.contains("hidden")) {
                closeAboutDialog();
            } else {
                openAboutDialog();
            }
        }
        // Escape closes About Dialog
        if (e.key === "Escape" && aboutModal && !aboutModal.classList.contains("hidden")) {
            e.preventDefault();
            closeAboutDialog();
        }
        // Repeat scan output (Spacebar)
        if (e.key === " " && e.target === document.body) {
            e.preventDefault();
            speakVoice(lastDetectedCurrency ? `Last scan: ${lastDetectedCurrency}` : "Scanning", true);
        }
    });

    // --- STATIC IMAGE UPLOAD HANDLING ---

    function handleImageUpload(e) {
        const file = e.target.files[0];
        if (!file) return;
        if (!file.type.startsWith("image/")) {
            announceStatus("Please select an image file.");
            speakVoice("Please select an image file.");
            imageUpload.value = "";
            return;
        }
        if (file.size > MAX_CLIENT_UPLOAD_BYTES) {
            announceStatus("Image is too large. Maximum size is 10 megabytes.");
            speakVoice("Image is too large.");
            imageUpload.value = "";
            return;
        }

        // 1. Pause WebSocket stream
        stopFrameStreaming();
        isProcessingFrame = false;

        // 2. Read image for preview
        const reader = new FileReader();
        reader.onload = (event) => {
            imagePreview.src = event.target.result;
            imagePreview.classList.remove("hidden");
            video.classList.add("hidden");
            resetCameraBtn.classList.remove("hidden");
        };
        reader.readAsDataURL(file);

        // 3. Prepare data to send to FastAPI POST REST API
        const formData = new FormData();
        formData.append("file", file);

        announceStatus("Uploading and recognizing image...");
        speakVoice("Analyzing image.");

        fetch("/api/detect", {
            method: "POST",
            body: formData
        })
        .then(async response => {
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) {
                throw new Error(payload.detail?.message || `Upload failed with status ${response.status}`);
            }
            return payload;
        })
        .then(data => {
            handleDetectionResult(data, "upload");
        })
        .catch(err => {
            console.error("Upload recognition failed:", err);
            const failureMessage = err.message || "Recognition failed.";
            announceStatus(failureMessage);
            speakVoice(failureMessage);
        });
    }

    function resetToLiveWebcam() {
        // 1. Hide image preview and button, show live video
        imagePreview.classList.add("hidden");
        imagePreview.src = "";
        video.classList.remove("hidden");
        resetCameraBtn.classList.add("hidden");
        if (bboxOverlay) bboxOverlay.classList.add("hidden");
        
        // Reset file input value
        imageUpload.value = "";

        // 2. Restart WebSocket stream
        announceStatus("Camera active. Align currency.");
        speakVoice("Returned to live scanning.");
        
        if (webSocket && webSocket.readyState === WebSocket.OPEN) {
            startFrameStreaming();
        } else {
            connectWebSocket();
        }
    }

    // About modal dialog controls
    function openAboutDialog() {
        if (!aboutModal) return;
        aboutModal.classList.remove("hidden");
        if (closeAboutBtn) closeAboutBtn.focus();
        speakVoice("About CurrenSight AI and Team dialog opened.");
    }

    function closeAboutDialog() {
        if (!aboutModal) return;
        aboutModal.classList.add("hidden");
        if (aboutBtn) aboutBtn.focus();
    }

    // Button event listeners
    contrastBtn.addEventListener("click", toggleContrastTheme);
    speechBtn.addEventListener("click", toggleSpeechEnabled);
    retryCameraBtn.addEventListener("click", startCamera);
    imageUpload.addEventListener("change", handleImageUpload);
    resetCameraBtn.addEventListener("click", resetToLiveWebcam);

    if (aboutBtn) aboutBtn.addEventListener("click", openAboutDialog);
    if (closeAboutBtn) closeAboutBtn.addEventListener("click", closeAboutDialog);
    if (dismissAboutBtn) dismissAboutBtn.addEventListener("click", closeAboutDialog);
    if (aboutModal) {
        aboutModal.addEventListener("click", (e) => {
            if (e.target === aboutModal) closeAboutDialog();
        });
    }

    document.addEventListener("visibilitychange", () => {
        if (document.hidden) {
            stopFrameStreaming();
        } else if (cameraStream?.active) {
            if (webSocket?.readyState === WebSocket.OPEN) startFrameStreaming();
            else connectWebSocket();
        }
    });

    window.addEventListener("beforeunload", () => {
        isShuttingDown = true;
        stopFrameStreaming();
        if (reconnectTimerId) clearTimeout(reconnectTimerId);
        if (webSocket && webSocket.readyState < WebSocket.CLOSING) webSocket.close(1000, "Page closed");
        if (cameraStream) cameraStream.getTracks().forEach(track => track.stop());
        window.speechSynthesis.cancel();
    });

    // Keyboard and Drag-and-Drop support for upload label
    const uploadLabel = document.getElementById("upload-label");
    if (uploadLabel) {
        uploadLabel.addEventListener("keydown", (e) => {
            if (e.key === " " || e.key === "Enter") {
                e.preventDefault();
                imageUpload.click();
            }
        });
        uploadLabel.addEventListener("dragover", (e) => {
            e.preventDefault();
            uploadLabel.classList.add("dragover");
        });
        uploadLabel.addEventListener("dragleave", () => {
            uploadLabel.classList.remove("dragover");
        });
        uploadLabel.addEventListener("drop", (e) => {
            e.preventDefault();
            uploadLabel.classList.remove("dragover");
            if (e.dataTransfer?.files?.length) {
                handleImageUpload({ target: { files: e.dataTransfer.files } });
            }
        });
    }

    // Initial Camera Activation
    startCamera();
});

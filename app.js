document.addEventListener('DOMContentLoaded', () => {
    // --- Navigation ---
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`${btn.dataset.tab}-tab`).classList.add('active');
        });
    });

    // --- System Health Check ---
    const statusBadge = document.getElementById('model-status-badge');
    const ageHint = document.getElementById('age-model-info');
    const genderHint = document.getElementById('gender-model-info');

    async function checkSystemHealth() {
        try {
            // We'll use a sample request or just a GET health check if we add one to Flask
            const resp = await fetch('http://localhost:5000/health');
            if (resp.ok) {
                statusBadge.textContent = "System Connected";
                statusBadge.className = "badge status-real";
            }
        } catch (e) {
            statusBadge.textContent = "Backend Offline";
            statusBadge.className = "badge status-demo";
        }
    }
    checkSystemHealth();

    // --- State & Shared Elements ---
    const loadingOverlay = document.querySelector('.loading-overlay');
    let audioBlob = null;
    let imageFile = null;

    // --- Voice Age Section ---
    const audioInput = document.getElementById('audio-input');
    const audioDropZone = document.getElementById('audio-drop-zone');
    const recordBtn = document.getElementById('record-btn');
    const recordStatus = document.getElementById('record-status');
    const audioPreview = document.getElementById('audio-preview');
    const audioPreviewContainer = document.getElementById('audio-preview-container');
    const clearAudioBtn = document.getElementById('clear-audio');
    const analyzeAgeBtn = document.getElementById('analyze-age-btn');
    const ageVal = document.getElementById('age-val');
    const ageResultArea = document.getElementById('age-result-area');

    let audioContext;
    let processor;
    let input;
    let stream;
    let leftchannel = [];
    let recordingLength = 0;
    let bufferSize = 2048;
    let sampleRate = 0;
    let isRecording = false;

    audioDropZone.addEventListener('click', () => audioInput.click());
    audioInput.addEventListener('change', (e) => handleAudioSelect(e.target.files[0]));
    recordBtn.addEventListener('click', toggleRecording);

    async function toggleRecording() {
        if (!isRecording) {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                sampleRate = audioContext.sampleRate;
                input = audioContext.createMediaStreamSource(stream);
                processor = audioContext.createScriptProcessor(bufferSize, 1, 1);

                leftchannel = [];
                recordingLength = 0;

                processor.onaudioprocess = function(e) {
                    leftchannel.push(new Float32Array(e.inputBuffer.getChannelData(0)));
                    recordingLength += bufferSize;
                };

                input.connect(processor);
                processor.connect(audioContext.destination);

                isRecording = true;
                recordBtn.classList.add('recording');
                recordStatus.textContent = "Recording... Click to stop";
            } catch (err) {
                alert("Microphone access denied: " + err);
            }
        } else {
            // STOP RECORDING
            processor.onaudioprocess = null;
            processor.disconnect();
            input.disconnect();
            stream.getTracks().forEach(track => track.stop());

            // Flatten channel
            let leftBuffer = flattenArray(leftchannel, recordingLength);
            // Create WAV
            let wavBuffer = createWav(leftBuffer);
            audioBlob = new Blob([wavBuffer], { type: 'audio/wav' });

            const url = URL.createObjectURL(audioBlob);
            audioPreview.src = url;
            audioPreviewContainer.classList.remove('hidden');
            analyzeAgeBtn.disabled = false;
            
            isRecording = false;
            recordBtn.classList.remove('recording');
            recordStatus.textContent = "Recording finished";
        }
    }

    function flattenArray(channelBuffer, recordingLength) {
        let result = new Float32Array(recordingLength);
        let offset = 0;
        for (let i = 0; i < channelBuffer.length; i++) {
            let buffer = channelBuffer[i];
            result.set(buffer, offset);
            offset += buffer.length;
        }
        return result;
    }

    function createWav(samples) {
        let buffer = new ArrayBuffer(44 + samples.length * 2);
        let view = new DataView(buffer);
        writeString(view, 0, 'RIFF');
        view.setUint32(4, 32 + samples.length * 2, true);
        writeString(view, 8, 'WAVE');
        writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(view, 36, 'data');
        view.setUint32(40, samples.length * 2, true);
        let index = 44;
        for (let i = 0; i < samples.length; i++) {
            let s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(index, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            index += 2;
        }
        return buffer;
    }

    function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    function handleAudioSelect(file) {
        if (file && file.type.startsWith('audio/')) {
            audioBlob = file;
            audioPreview.src = URL.createObjectURL(file);
            audioPreviewContainer.classList.remove('hidden');
            analyzeAgeBtn.disabled = false;
            ageResultArea.classList.add('hidden');
        }
    }

    clearAudioBtn.addEventListener('click', () => {
        audioBlob = null;
        audioPreview.src = "";
        audioPreviewContainer.classList.add('hidden');
        analyzeAgeBtn.disabled = true;
        ageResultArea.classList.add('hidden');
        audioInput.value = "";
    });

    analyzeAgeBtn.addEventListener('click', async () => {
        if (!audioBlob) return;
        loadingOverlay.classList.remove('hidden');
        
        const formData = new FormData();
        formData.append('audio', audioBlob, 'sample.wav');

        try {
            const resp = await fetch('http://localhost:5000/predict_age', {
                method: 'POST',
                body: formData
            });
            const data = await resp.json();
            if (data.success) {
                ageVal.textContent = data.age_group;
                ageResultArea.classList.remove('hidden');
                ageHint.textContent = `Mode: ${data.model_used}`;
                
                // Show confidence
                const confText = document.getElementById('age-conf-text');
                if (data.confidence !== undefined) {
                    confText.textContent = `Model Confidence: ${data.confidence}%`;
                } else if (data.estimated_age) {
                    confText.textContent = `Estimated Age: ${data.estimated_age}`;
                }
                
                // Show pitch info
                const pitchInfo = document.getElementById('pitch-info');
                let pitchText = '';
                if (data.pitch_score) {
                    pitchText = `🎵 Detected Pitch: ${data.pitch_score} Hz`;
                    if (data.pitch_group) pitchText += `  →  Pitch suggests: ${data.pitch_group}`;
                }
                if (data.correction_applied) {
                    pitchText += `\n⚡ Hybrid correction applied (model was uncertain at ${data.confidence}%)`;
                    pitchInfo.style.color = '#f59e0b';
                } else {
                    pitchInfo.style.color = '';
                }
                pitchInfo.textContent = pitchText;
                
                // Show per-class score bars
                const scoresDiv = document.getElementById('scores-bars');
                const breakdownDiv = document.getElementById('age-scores-breakdown');
                if (data.all_scores) {
                    scoresDiv.innerHTML = '';
                    for (const [group, score] of Object.entries(data.all_scores)) {
                        const isWinner = group === data.age_group;
                        scoresDiv.innerHTML += `
                            <div class="score-row ${isWinner ? 'winner' : ''}">
                                <span class="score-label">${group}</span>
                                <div class="score-track">
                                    <div class="score-fill" style="width: ${score}%"></div>
                                </div>
                                <span class="score-pct">${score}%</span>
                            </div>`;
                    }
                    breakdownDiv.classList.remove('hidden');
                } else {
                    breakdownDiv.classList.add('hidden');
                }
            } else {
                alert("Error: " + data.error);
            }
        } catch (err) {
            alert("API connection failed. Ensure backend is running.");
        } finally {
            loadingOverlay.classList.add('hidden');
        }
    });

    // --- Fingerprint Gender Section ---
    const imageInput = document.getElementById('image-input');
    const imageDropZone = document.getElementById('image-drop-zone');
    const imagePreview = document.getElementById('image-preview');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const clearImageBtn = document.getElementById('clear-image');
    const analyzeGenderBtn = document.getElementById('analyze-gender-btn');
    const genderVal = document.getElementById('gender-val');
    const genderResultArea = document.getElementById('gender-result-area');
    const genderConfText = document.getElementById('gender-conf-text');

    imageDropZone.addEventListener('click', () => imageInput.click());
    imageInput.addEventListener('change', (e) => handleImageSelect(e.target.files[0]));

    function handleImageSelect(file) {
        if (file && file.type.startsWith('image/')) {
            imageFile = file;
            imagePreview.src = URL.createObjectURL(file);
            imagePreviewContainer.classList.remove('hidden');
            analyzeGenderBtn.disabled = false;
            genderResultArea.classList.add('hidden');
        }
    }

    clearImageBtn.addEventListener('click', () => {
        imageFile = null;
        imagePreview.src = "";
        imagePreviewContainer.classList.add('hidden');
        analyzeGenderBtn.disabled = true;
        genderResultArea.classList.add('hidden');
        imageInput.value = "";
    });

    analyzeGenderBtn.addEventListener('click', async () => {
        if (!imageFile) return;
        loadingOverlay.classList.remove('hidden');

        const formData = new FormData();
        formData.append('image', imageFile);

        try {
            const resp = await fetch('http://localhost:5000/predict_gender', {
                method: 'POST',
                body: formData
            });
            const data = await resp.json();
            if (data.success) {
                genderVal.textContent = data.gender;
                genderConfText.textContent = `Confidence: ${data.confidence}%`;
                genderResultArea.classList.remove('hidden');
                genderHint.textContent = `Mode: ${data.model_used}`;
            } else {
                alert("Error: " + data.error);
            }
        } catch (err) {
            alert("API connection failed.");
        } finally {
            loadingOverlay.classList.add('hidden');
        }
    });
});

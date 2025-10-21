// Global state
let ws, audioContext, processor, source, stream;
let isRecording = false;
let timerInterval;
let startTime;
let audioBuffer = new Int16Array(0);
let wsConnected = false;
let streamInitialized = false;
let isAutoStarted = false;

// DOM elements
const recordButton = document.getElementById('recordButton');
const transcript = document.getElementById('transcript');
const enhancedTranscript = document.getElementById('enhancedTranscript');
const copyButton = document.getElementById('copyButton');
const copyEnhancedButton = document.getElementById('copyEnhancedButton');
const readabilityButton = document.getElementById('readabilityButton');
const askAIButton = document.getElementById('askAIButton');
const correctnessButton = document.getElementById('correctnessButton');
const apiKeyInput = document.getElementById('apiKeyInput');
const saveApiKeyButton = document.getElementById('saveApiKeyButton');
const settingsButton = document.getElementById('settingsButton');
const settingsModal = document.getElementById('settingsModal');
const closeModal = document.getElementById('closeModal');
const apiKeyStatusText = document.getElementById('apiKeyStatusText');
const llmProviderSelect = document.getElementById('llmProviderSelect');
const openaiSettingsSection = document.getElementById('openaiSettingsSection');
const compatibleSettingsSection = document.getElementById('compatibleSettingsSection');
const openaiModelSelect = document.getElementById('openaiModelSelect');
const compatibleBaseUrlInput = document.getElementById('compatibleBaseUrlInput');
const compatibleModelInput = document.getElementById('compatibleModelInput');
const compatibleApiKeyInput = document.getElementById('compatibleApiKeyInput');
const saveCompatibleButton = document.getElementById('saveCompatibleButton');
const compatibleStatusText = document.getElementById('compatibleStatusText');

// Configuration
const targetSeconds = 5;
const urlParams = new URLSearchParams(window.location.search);
const autoStart = urlParams.get('start') === '1';

// Utility functions
const isMobileDevice = () => /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

// API Key management
let userApiKey = null;
let openaiModel = 'gpt-4o';
let compatibleBaseUrl = null;
let compatibleModel = null;
let compatibleApiKey = null;
let llmProvider = 'compatible'; // Default to compatible (for GLM)

function openSettingsModal() {
    settingsModal.classList.add('active');
    apiKeyInput.focus();
}

function closeSettingsModal() {
    settingsModal.classList.remove('active');
}

function updateApiKeyStatus() {
    if (userApiKey) {
        apiKeyStatusText.textContent = 'Saved';
        apiKeyStatusText.classList.add('saved');
    } else {
        apiKeyStatusText.textContent = '';
        apiKeyStatusText.classList.remove('saved');
    }
}

function updateCompatibleStatus() {
    if (compatibleBaseUrl && compatibleModel && compatibleApiKey) {
        compatibleStatusText.textContent = 'Saved';
        compatibleStatusText.classList.add('saved');
    } else {
        compatibleStatusText.textContent = '';
        compatibleStatusText.classList.remove('saved');
    }
}

function loadApiKey() {
    // Load OpenAI settings
    const savedKey = localStorage.getItem('openai_api_key');
    if (savedKey) {
        userApiKey = savedKey;
        apiKeyInput.value = savedKey;
        updateApiKeyStatus();
    }

    const savedModel = localStorage.getItem('openai_model');
    if (savedModel) {
        openaiModel = savedModel;
        openaiModelSelect.value = savedModel;
    }

    // Load Compatible settings
    const savedBaseUrl = localStorage.getItem('compatible_base_url');
    const savedCompatibleModel = localStorage.getItem('compatible_model');
    const savedCompatibleKey = localStorage.getItem('compatible_api_key');

    if (savedBaseUrl) {
        compatibleBaseUrl = savedBaseUrl;
        compatibleBaseUrlInput.value = savedBaseUrl;
    }
    if (savedCompatibleModel) {
        compatibleModel = savedCompatibleModel;
        compatibleModelInput.value = savedCompatibleModel;
    }
    if (savedCompatibleKey) {
        compatibleApiKey = savedCompatibleKey;
        compatibleApiKeyInput.value = savedCompatibleKey;
    }
    updateCompatibleStatus();

    // Load saved LLM provider preference
    const savedProvider = localStorage.getItem('llm_provider');
    if (savedProvider) {
        llmProvider = savedProvider;
        llmProviderSelect.value = savedProvider;
    }

    // Show/hide sections based on provider
    updateSettingsVisibility();
}

function updateSettingsVisibility() {
    if (llmProvider === 'compatible') {
        compatibleSettingsSection.style.display = 'block';
        openaiSettingsSection.style.display = 'none';
    } else {
        compatibleSettingsSection.style.display = 'none';
        openaiSettingsSection.style.display = 'block';
    }
}

function saveLLMProvider() {
    const provider = llmProviderSelect.value;
    llmProvider = provider;
    localStorage.setItem('llm_provider', provider);
    updateSettingsVisibility();
}

function saveApiKey() {
    const apiKey = apiKeyInput.value.trim();
    const model = openaiModelSelect.value;

    if (!apiKey) {
        alert('Please enter an API key');
        return;
    }

    if (!apiKey.startsWith('sk-')) {
        alert('Invalid API key format. OpenAI API keys start with "sk-"');
        return;
    }

    userApiKey = apiKey;
    openaiModel = model;
    localStorage.setItem('openai_api_key', apiKey);
    localStorage.setItem('openai_model', model);

    // Send API key to server via WebSocket
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'set_api_key',
            api_key: apiKey
        }));
    }

    updateApiKeyStatus();
    showApiKeyFeedback('Saved!');

    // Close modal after a short delay
    setTimeout(() => {
        closeSettingsModal();
    }, 1000);
}

function saveCompatibleSettings() {
    const baseUrl = compatibleBaseUrlInput.value.trim();
    const model = compatibleModelInput.value.trim();
    const apiKey = compatibleApiKeyInput.value.trim();

    if (!baseUrl) {
        alert('Please enter base URL');
        return;
    }

    if (!model) {
        alert('Please enter model name');
        return;
    }

    if (!apiKey) {
        alert('Please enter API key');
        return;
    }

    compatibleBaseUrl = baseUrl;
    compatibleModel = model;
    compatibleApiKey = apiKey;
    localStorage.setItem('compatible_base_url', baseUrl);
    localStorage.setItem('compatible_model', model);
    localStorage.setItem('compatible_api_key', apiKey);

    updateCompatibleStatus();
    showCompatibleFeedback('Saved!');

    // Close modal after a short delay
    setTimeout(() => {
        closeSettingsModal();
    }, 1000);
}

function showApiKeyFeedback(message) {
    const originalText = saveApiKeyButton.textContent;
    saveApiKeyButton.textContent = message;
    setTimeout(() => {
        saveApiKeyButton.textContent = originalText;
    }, 2000);
}

function showCompatibleFeedback(message) {
    const originalText = saveCompatibleButton.textContent;
    saveCompatibleButton.textContent = message;
    setTimeout(() => {
        saveCompatibleButton.textContent = originalText;
    }, 2000);
}

function getApiKey() {
    return userApiKey || null;
}

async function copyToClipboard(text, button) {
    if (!text) return;
    try {
        await navigator.clipboard.writeText(text);
        showCopiedFeedback(button, 'Copied!');
    } catch (err) {
        console.error('Clipboard copy failed:', err);
        // alert('Clipboard copy failed: ' + err.message);
        // We don't show this message because it's not accurate. We could still write to the clipboard in this case.
    }
}

function showCopiedFeedback(button, message) {
    if (!button) return;
    const originalText = button.textContent;
    button.textContent = message;
    setTimeout(() => {
        button.textContent = originalText;
    }, 2000);
}

// Timer functions
function startTimer() {
    clearInterval(timerInterval);
    document.getElementById('timer').textContent = '00:00';
    startTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Date.now() - startTime;
        const minutes = Math.floor(elapsed / 60000);
        const seconds = Math.floor((elapsed % 60000) / 1000);
        document.getElementById('timer').textContent = 
            `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }, 1000);
}

function stopTimer() {
    clearInterval(timerInterval);
}

// Audio processing
function createAudioProcessor() {
    processor = audioContext.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (e) => {
        if (!isRecording) return;
        
        const inputData = e.inputBuffer.getChannelData(0);
        const pcmData = new Int16Array(inputData.length);
        
        for (let i = 0; i < inputData.length; i++) {
            pcmData[i] = Math.max(-32768, Math.min(32767, Math.floor(inputData[i] * 32767)));
        }
        
        const combinedBuffer = new Int16Array(audioBuffer.length + pcmData.length);
        combinedBuffer.set(audioBuffer);
        combinedBuffer.set(pcmData, audioBuffer.length);
        audioBuffer = combinedBuffer;
        
        if (audioBuffer.length >= 24000) {
            const sendBuffer = audioBuffer.slice(0, 24000);
            audioBuffer = audioBuffer.slice(24000);
            
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(sendBuffer.buffer);
            }
        }
    };
    return processor;
}

async function initAudio(stream) {
    audioContext = new AudioContext();
    source = audioContext.createMediaStreamSource(stream);
    processor = createAudioProcessor();
    source.connect(processor);
    processor.connect(audioContext.destination);
}

// WebSocket handling
function updateConnectionStatus(status) {
    const statusDot = document.getElementById('connectionStatus');
    if (!statusDot) return; // Element removed, skip status updates

    statusDot.classList.remove('connected', 'connecting', 'idle');

    switch (status) {
        case 'connected':  // OpenAI is connected and ready
            statusDot.classList.add('connected');
            statusDot.style.backgroundColor = '#34C759';  // Green
            break;
        case 'connecting':  // Establishing OpenAI connection
            statusDot.classList.add('connecting');
            statusDot.style.backgroundColor = '#FF9500';  // Orange
            break;
        case 'idle':  // Client connected, OpenAI not connected
            statusDot.classList.add('idle');
            statusDot.style.backgroundColor = '#007AFF';  // Blue
            break;
        default:  // Disconnected
            statusDot.style.backgroundColor = '#FF3B30';  // Red
    }
}

function initializeWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${protocol}://${window.location.host}/api/v1/ws`);

    ws.onopen = () => {
        wsConnected = true;
        updateConnectionStatus(true);

        // Send API key to server if available
        if (userApiKey) {
            ws.send(JSON.stringify({
                type: 'set_api_key',
                api_key: userApiKey
            }));
        }

        if (autoStart && !isRecording && !isAutoStarted) startRecording();
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        switch (data.type) {
            case 'status':
                updateConnectionStatus(data.status);
                if (data.status === 'idle') {
                    copyToClipboard(transcript.value, copyButton);
                }
                break;
            case 'text':
                if (data.isNewResponse) {
                    transcript.value = data.content;
                    stopTimer();
                } else {
                    transcript.value += data.content;
                }
                transcript.scrollTop = transcript.scrollHeight;
                break;
            case 'error':
                alert(data.content);
                updateConnectionStatus('idle');
                break;
        }
    };
    
    ws.onclose = () => {
        wsConnected = false;
        updateConnectionStatus(false);
        setTimeout(initializeWebSocket, 1000);
    };
}

// Recording control
async function startRecording() {
    if (isRecording) return;

    // Check if OpenAI API key is set
    if (!userApiKey) {
        alert('Please set your OpenAI API key in Settings before starting recording.');
        openSettingsModal();
        return;
    }

    try {
        transcript.value = '';
        enhancedTranscript.value = '';

        if (!streamInitialized) {
            // Check if getUserMedia is supported
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('Your browser does not support microphone access. Please use HTTPS or localhost, and ensure you are using a modern browser.');
            }

            stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            streamInitialized = true;
        }

        if (!stream) throw new Error('Failed to initialize audio stream');
        if (!audioContext) await initAudio(stream);

        isRecording = true;
        await ws.send(JSON.stringify({ type: 'start_recording' }));

        startTimer();
        recordButton.textContent = 'Stop';
        recordButton.classList.add('recording');

    } catch (error) {
        console.error('Error starting recording:', error);
        alert('Error accessing microphone: ' + error.message);
    }
}

async function stopRecording() {
    if (!isRecording) return;
    
    isRecording = false;
    startTimer();
    
    if (audioBuffer.length > 0 && ws.readyState === WebSocket.OPEN) {
        ws.send(audioBuffer.buffer);
        audioBuffer = new Int16Array(0);
    }
    
    await new Promise(resolve => setTimeout(resolve, 500));
    await ws.send(JSON.stringify({ type: 'stop_recording' }));
    
    recordButton.textContent = 'Start';
    recordButton.classList.remove('recording');
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Event listeners
    recordButton.onclick = () => isRecording ? stopRecording() : startRecording();
    copyButton.onclick = () => copyToClipboard(transcript.value, copyButton);
    copyEnhancedButton.onclick = () => copyToClipboard(enhancedTranscript.value, copyEnhancedButton);
    saveApiKeyButton.onclick = saveApiKey;
    saveCompatibleButton.onclick = saveCompatibleSettings;
    settingsButton.onclick = openSettingsModal;
    closeModal.onclick = closeSettingsModal;
    llmProviderSelect.onchange = saveLLMProvider;

    // Close modal when clicking outside
    settingsModal.onclick = (e) => {
        if (e.target === settingsModal) {
            closeSettingsModal();
        }
    };

    // Close modal with Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && settingsModal.classList.contains('active')) {
            closeSettingsModal();
        }
    });

    // Allow Enter key to save API key
    apiKeyInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            saveApiKey();
        }
    });

    // Handle spacebar toggle
    document.addEventListener('keydown', (event) => {
        if (event.code === 'Space') {
            const activeElement = document.activeElement;
            const modalOpen = settingsModal.classList.contains('active');
            if (!activeElement.tagName.match(/INPUT|TEXTAREA/) && !activeElement.isContentEditable && !modalOpen) {
                event.preventDefault();
                recordButton.click();
            }
        }
    });

    // Readability and AI handlers
    readabilityButton.onclick = async () => {
    startTimer();
    const inputText = transcript.value.trim();
    if (!inputText) {
        alert('Please enter text to enhance readability.');
        stopTimer();
        return;
    }

    try {
        const apiKey = getApiKey();
        const requestBody = {
            text: inputText,
            llm_provider: llmProvider
        };

        if (llmProvider === 'openai') {
            requestBody.api_key = apiKey;
            requestBody.openai_model = openaiModel || 'gpt-4o';
        } else {
            requestBody.compatible_base_url = compatibleBaseUrl || '';
            requestBody.compatible_model = compatibleModel || '';
            requestBody.compatible_api_key = compatibleApiKey || '';
        }

        const response = await fetch('/api/v1/readability', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) throw new Error('Readability enhancement failed');

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            fullText += decoder.decode(value, { stream: true });
            enhancedTranscript.value = fullText;
            enhancedTranscript.scrollTop = enhancedTranscript.scrollHeight;
        }

        if (!isMobileDevice()) copyToClipboard(fullText, copyEnhancedButton);
        stopTimer();

    } catch (error) {
        console.error('Error:', error);
        alert('Error enhancing readability');
        stopTimer();
    }
    };

    askAIButton.onclick = async () => {
    startTimer();
    const inputText = transcript.value.trim();
    if (!inputText) {
        alert('Please enter text to ask AI about.');
        stopTimer();
        return;
    }

    try {
        const apiKey = getApiKey();
        const requestBody = {
            text: inputText,
            llm_provider: llmProvider
        };

        if (llmProvider === 'openai') {
            requestBody.api_key = apiKey;
            requestBody.openai_model = openaiModel || 'gpt-4o';
        } else {
            requestBody.compatible_base_url = compatibleBaseUrl || '';
            requestBody.compatible_model = compatibleModel || '';
            requestBody.compatible_api_key = compatibleApiKey || '';
        }

        const response = await fetch('/api/v1/ask_ai', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) throw new Error('AI request failed');

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            fullText += decoder.decode(value, { stream: true });
            enhancedTranscript.value = fullText;
            enhancedTranscript.scrollTop = enhancedTranscript.scrollHeight;
        }

        if (!isMobileDevice()) copyToClipboard(fullText, copyEnhancedButton);
        stopTimer();

    } catch (error) {
        console.error('Error:', error);
        alert('Error asking AI');
        stopTimer();
    }
    };

    correctnessButton.onclick = async () => {
    startTimer();
    const inputText = transcript.value.trim();
    if (!inputText) {
        alert('Please enter text to check for correctness.');
        stopTimer();
        return;
    }

    try {
        const apiKey = getApiKey();
        const requestBody = {
            text: inputText,
            llm_provider: llmProvider
        };

        if (llmProvider === 'openai') {
            requestBody.api_key = apiKey;
            requestBody.openai_model = openaiModel || 'gpt-4o';
        } else {
            requestBody.compatible_base_url = compatibleBaseUrl || '';
            requestBody.compatible_model = compatibleModel || '';
            requestBody.compatible_api_key = compatibleApiKey || '';
        }

        const response = await fetch('/api/v1/correctness', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) throw new Error('Correctness check failed');

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            fullText += decoder.decode(value, { stream: true });
            enhancedTranscript.value = fullText;
            enhancedTranscript.scrollTop = enhancedTranscript.scrollHeight;
        }

        if (!isMobileDevice()) copyToClipboard(fullText, copyEnhancedButton);
        stopTimer();

    } catch (error) {
        console.error('Error:', error);
        alert('Error checking correctness');
        stopTimer();
    }
    };

    // Theme toggle
    document.getElementById('themeToggle').onclick = toggleTheme;

    // Initialize
    loadApiKey();
    initializeWebSocket();
    initializeTheme();
    if (autoStart) initializeAudioStream();
});

// Theme handling
function toggleTheme() {
    const body = document.body;
    const themeToggle = document.getElementById('themeToggle');
    const isDarkTheme = body.classList.toggle('dark-theme');
    
    // Update button text
    themeToggle.textContent = isDarkTheme ? '☀️' : '🌙';
    
    // Save preference to localStorage
    localStorage.setItem('darkTheme', isDarkTheme);
}

// Initialize theme from saved preference
function initializeTheme() {
    const darkTheme = localStorage.getItem('darkTheme') === 'true';
    const themeToggle = document.getElementById('themeToggle');

    if (darkTheme) {
        document.body.classList.add('dark-theme');
        themeToggle.textContent = '☀️';
    }
}

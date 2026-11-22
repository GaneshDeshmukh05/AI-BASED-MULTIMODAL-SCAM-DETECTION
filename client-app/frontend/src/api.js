
import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const api = axios.create({
    baseURL: API_URL,
    timeout: 120000,
});

// Check backend connection
export const checkBackendHealth = async () => {
    const response = await api.get("/api/health");
    return response.data;
};

// Submit a new multimodal scan
export const scanMultimodal = async ({
    text,
    url,
    audioFile,
    screenshotFile,
    offlineMode = true,
}) => {
    const formData = new FormData();

    if (text?.trim()) {
        formData.append("text", text.trim());
    }

    if (url?.trim()) {
        formData.append("url", url.trim());
    }

    if (audioFile) {
        formData.append("audio_file", audioFile);
    }

    if (screenshotFile) {
        formData.append("screenshot_file", screenshotFile);
    }

    formData.append("offline_mode", String(offlineMode));

    const response = await api.post("/api/scan", formData);
    return response.data;
};

// Retrieve saved scan history
export const getScanHistory = async (limit = 100) => {
    const response = await api.get("/api/history", {
        params: { limit },
    });

    return response.data;
};

// Retrieve details of an individual scan
export const getScanDetails = async (scanId) => {
    const response = await api.get(
        `/api/history/${encodeURIComponent(scanId)}`
    );

    return response.data;
};

export default api;

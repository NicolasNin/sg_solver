/**
 * Star Genius - Camera Module
 * 
 * Provides camera access with overlay guide for mobile board capture.
 */

const StarGeniusCamera = (function () {
    let videoStream = null;
    let isOpen = false;

    /**
     * Check if device supports camera access
     */
    function isSupported() {
        return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    }

    /**
     * Check if we're likely on a mobile device
     */
    function isMobile() {
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    }

    /**
     * Open the camera modal and start video stream
     */
    async function open() {
        if (!isSupported()) {
            alert('Camera not supported on this device/browser');
            return false;
        }

        const modal = document.getElementById('camera-modal');
        const video = document.getElementById('camera-video');
        const errorEl = document.getElementById('camera-error');

        if (!modal || !video) {
            console.error('Camera modal elements not found');
            return false;
        }

        // Show modal immediately
        modal.classList.remove('hidden');
        errorEl.classList.add('hidden');

        try {
            // Request camera access - prefer rear camera on mobile
            const constraints = {
                video: {
                    facingMode: { ideal: 'environment' },  // Rear camera
                    width: { ideal: 1920 },
                    height: { ideal: 1080 }
                },
                audio: false
            };

            videoStream = await navigator.mediaDevices.getUserMedia(constraints);
            video.srcObject = videoStream;

            // Wait for video to be ready
            await new Promise((resolve) => {
                video.onloadedmetadata = () => {
                    video.play();
                    resolve();
                };
            });

            isOpen = true;
            return true;

        } catch (error) {
            console.error('Camera access error:', error);
            errorEl.textContent = getErrorMessage(error);
            errorEl.classList.remove('hidden');
            return false;
        }
    }

    /**
     * Get user-friendly error message
     */
    function getErrorMessage(error) {
        switch (error.name) {
            case 'NotAllowedError':
                return '📷 Camera permission denied. Please allow camera access and try again.';
            case 'NotFoundError':
                return '📷 No camera found on this device.';
            case 'NotReadableError':
                return '📷 Camera is in use by another application.';
            case 'OverconstrainedError':
                return '📷 Camera does not support required settings.';
            default:
                return `📷 Camera error: ${error.message}`;
        }
    }

    /**
     * Close camera modal and stop stream
     */
    function close() {
        const modal = document.getElementById('camera-modal');
        const video = document.getElementById('camera-video');

        // Stop all video tracks
        if (videoStream) {
            videoStream.getTracks().forEach(track => track.stop());
            videoStream = null;
        }

        // Clear video source
        if (video) {
            video.srcObject = null;
        }

        // Hide modal
        if (modal) {
            modal.classList.add('hidden');
        }

        isOpen = false;
    }

    /**
     * Capture current video frame and return as Blob
     */
    async function capture() {
        const video = document.getElementById('camera-video');

        if (!video || !videoStream) {
            console.error('No video stream available');
            return null;
        }

        // Create canvas with video dimensions
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);

        // Convert to blob
        return new Promise((resolve) => {
            canvas.toBlob((blob) => {
                resolve(blob);
            }, 'image/jpeg', 0.9);
        });
    }

    /**
     * Capture and return as File (for existing loadFromPhoto API)
     */
    async function captureAsFile() {
        const blob = await capture();
        if (!blob) return null;

        // Create File object from blob
        const filename = `capture_${Date.now()}.jpg`;
        return new File([blob], filename, { type: 'image/jpeg' });
    }

    // Public API
    return {
        isSupported,
        isMobile,
        open,
        close,
        capture,
        captureAsFile,
        get isOpen() { return isOpen; }
    };
})();

// Export to window
window.StarGeniusCamera = StarGeniusCamera;

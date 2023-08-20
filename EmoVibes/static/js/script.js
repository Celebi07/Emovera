const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');

navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        video.srcObject = stream;
        video.play();
    })
    .catch((error) => {
        console.error('Error accessing webcam:', error);
    });

video.addEventListener('play', () => {
    canvas.width = video.width;
    canvas.height = video.height;

    const socket = io(); // Initialize Socket.IO connection (make sure to include Socket.IO library in your HTML)

    function detectEmotion() {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert canvas image to base64 data URL
        const imageData = canvas.toDataURL('image/jpeg', 0.7); // Adjust quality if needed

        // Send captured image to Flask backend for emotion detection
        socket.emit('image', { data: imageData });

        requestAnimationFrame(detectEmotion);
    }

    detectEmotion();
});

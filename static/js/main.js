let audioCtx;
let oscillator;
let gainNode;
// Declare detectionInterval variable to clear the interval later
let detectionInterval;

function initAudioContext() {
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  oscillator = audioCtx.createOscillator();
  gainNode = audioCtx.createGain();

  oscillator.connect(gainNode);
  gainNode.connect(audioCtx.destination);
  oscillator.type = 'sine';
  oscillator.frequency.setValueAtTime(440, audioCtx.currentTime); // 440 Hz frequency (A4 note)
  gainNode.gain.setValueAtTime(0, audioCtx.currentTime); // Start with 0 gain to make it inaudible
}

function playAlarm() {
  if (audioCtx.state === 'suspended') {
    audioCtx.resume();
  }

  gainNode.gain.setValueAtTime(0, audioCtx.currentTime);
  gainNode.gain.linearRampToValueAtTime(1, audioCtx.currentTime + 0.01); // Increasing gain to 1 in 0.01 seconds
  gainNode.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 0.3); // Decreasing gain to 0 in 0.3 seconds

  oscillator.start(audioCtx.currentTime);
  oscillator.stop(audioCtx.currentTime + 0.5); // Stop playing after 0.5 seconds
}

// Call this function once to initialize the audio context.
initAudioContext();

async function setupWebcam() {
  const video = document.getElementById("videoFeed");
  navigator.mediaDevices.getUserMedia = navigator.mediaDevices.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;

  const stream = await navigator.mediaDevices.getUserMedia({ video: {} });
  video.srcObject = stream;

  // Make the live feed visible
  video.classList.remove("d-none");

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadModels() {
  await faceapi.nets.faceLandmark68TinyNet.loadFromUri('/models/face_landmark_68_tiny_model');
  await faceapi.nets.tinyFaceDetector.loadFromUri('/models/tiny_face_detector_model');
}

async function calculateEAR(landmarks) {
  const leftEye = landmarks.getLeftEye();
  const rightEye = landmarks.getRightEye();
  
  const leftEAR = eye_aspect_ratio(leftEye);
  const rightEAR = eye_aspect_ratio(rightEye);
  
  const ear = (leftEAR + rightEAR) / 2;
  return ear;
}

function eye_aspect_ratio(eye) {
  const A = distance(eye[1], eye[5]);
  const B = distance(eye[2], eye[4]);
  const C = distance(eye[0], eye[3]);
  return (A + B) / (2.0 * C);
}

function distance(point1, point2) {
  return Math.sqrt(Math.pow(point2.x - point1.x, 2) + Math.pow(point2.y - point1.y, 2));
}

async function main() {
  await loadModels();
  const video = await setupWebcam();

  const canvas = faceapi.createCanvasFromMedia(video);
  document.body.append(canvas);

  const displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);

  detectionInterval = setInterval(async () => {
    const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks();

    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);

    if (resizedDetections && resizedDetections.length > 0) {
      const ear = await calculateEAR(resizedDetections[0].landmarks);
      if (ear < 0.15) {
        playAlarm();
      }
    }

    faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
  }, 100);
}

main();

let session;

// Load ONNX model
async function initModel() {
  session = await ort.InferenceSession.create('model/cnn_model.onnx');
  console.log('Model loaded');
}

// Preprocess uploaded image to [1, 3, 224, 224] float32
function preprocessImage(imgElement) {
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');

  // Resize to 224x224
  canvas.width = 224;
  canvas.height = 224;
  ctx.drawImage(imgElement, 0, 0, 224, 224);

  // Get pixel data and normalize to [0,1]
  const imageData = ctx.getImageData(0, 0, 224, 224).data;
  const float32Data = new Float32Array(3 * 224 * 224);

  for (let i = 0; i < 224*224; i++) {
    float32Data[i] = imageData[i*4] / 255.0;          // R
    float32Data[i + 224*224] = imageData[i*4 + 1] / 255.0; // G
    float32Data[i + 2*224*224] = imageData[i*4 + 2] / 255.0; // B
  }

  return new ort.Tensor('float32', float32Data, [1, 3, 224, 224]);
}

// Run model on uploaded image
async function runModel() {
  const fileInput = document.getElementById('imageInput');
  if (fileInput.files.length === 0) {
    alert("Please upload an image first.");
    return;
  }

  const img = new Image();
  img.src = URL.createObjectURL(fileInput.files[0]);
  img.onload = async () => {
    const inputTensor = preprocessImage(img);
    const feeds = { input: inputTensor };
    const results = await session.run(feeds);

    const output = results.output.data;
    const classIndex = output[0] > output[1] ? 0 : 1;
    const classes = ["Cat", "Dog"];
    document.getElementById('output').innerText = `Prediction: ${classes[classIndex]}`;
  };
}

// Initialize model on page load
initModel();

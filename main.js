let session;

// Load ONNX model
async function initModel() {
  session = await ort.InferenceSession.create('cnn_model_merged.onnx');
  console.log('Model loaded');
  document.getElementById('classifyBtn').disabled = false;
}

// Preprocess uploaded image to [1, 3, 224, 224] float32
function preprocessImage(imgElement) {
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');

  // Resize to 224x224
  canvas.width = 64;
  canvas.height = 64;
  ctx.drawImage(imgElement, 0, 0, 64, 64);

  // Get pixel data and normalize to [0,1]
  const imageData = ctx.getImageData(0, 0, 64, 64).data;
  console.log('ImageData length:', imageData.length);
  const float32Data = new Float32Array(3 * 64 * 64);

  for (let i = 0; i < 64*64; i++) {
    float32Data[i] = (imageData[i*4]/255.0 - 0.5)/0.5;       // R
    float32Data[i + 64*64] = (imageData[i*4+1]/255.0 - 0.5)/0.5; // G
    float32Data[i + 2*64*64] = (imageData[i*4+2]/255.0 - 0.5)/0.5; // B
  }

  return new ort.Tensor('float32', float32Data, [1, 3, 64, 64]);
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
      const feeds = { "input": inputTensor }; // exact tensor name from ONNX export
      const results = await session.run(feeds);
  
      const output = results.output.data; // raw logits from model
      const exp0 = Math.exp(output[0]);
      const exp1 = Math.exp(output[1]);
      const sumExp = exp0 + exp1;

      const prob0 = exp0 / sumExp; // probability for class 0
      const prob1 = exp1 / sumExp; // probability for class 1

      const classIndex = prob0 > prob1 ? 0 : 1;
      const classes = ["Cat", "Dog"];
      document.getElementById('output').innerText =
        `Prediction: ${classes[classIndex]} \n` +
        `Probabilities: Cat: ${prob0.toFixed(3)}, Dog: ${prob1.toFixed(3)}`;
  };
}

// Initialize model on page load
initModel();

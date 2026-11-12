/**
 * DeepPlate / PlateGuard – Express API Server
 *
 * Routes:
 *   GET  /               → health ping
 *   GET  /api/health     → server + ML service status
 *   POST /api/predict    → proxy image upload to FastAPI ML service
 */

const express = require('express');
const cors = require('cors');
const multer = require('multer');
require('dotenv').config();

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://127.0.0.1:8000';

app.use(cors());
app.use(express.json());

/* ── Basic health ──────────────────────────────────────────────────────── */
app.get('/', (_req, res) => {
  res.json({ status: 'ok', service: 'DeepPlate API server' });
});

/* ── Combined health (pings ML service) ───────────────────────────────── */
app.get('/api/health', async (_req, res) => {
  let mlHealth = { status: 'unreachable' };
  try {
    const r = await fetch(`${ML_SERVICE_URL}/health`, { signal: AbortSignal.timeout(3000) });
    mlHealth = await r.json();
  } catch {
    // ml service is down or not started – not a fatal server error
  }
  res.json({ server: 'ok', ml_service: mlHealth });
});

/* ── Predict – proxy to FastAPI ────────────────────────────────────────── */
app.post('/api/predict', upload.single('file'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ detail: 'No file uploaded. Send a multipart field named "file".' });
  }

  const form = new FormData();
  form.append(
    'file',
    new Blob([req.file.buffer], { type: req.file.mimetype }),
    req.file.originalname || 'upload.jpg',
  );

  try {
    const mlRes = await fetch(`${ML_SERVICE_URL}/api/predict`, {
      method: 'POST',
      body: form,
      signal: AbortSignal.timeout(30_000),
    });

    const data = await mlRes.json();

    if (!mlRes.ok) {
      return res.status(mlRes.status).json(data);
    }

    return res.json(data);
  } catch (err) {
    console.error('[/api/predict] ML service error:', err.message);
    return res.status(502).json({
      detail: 'ML service unavailable. Make sure the Python FastAPI server is running on port 8000.',
    });
  }
});

/* ── Start ─────────────────────────────────────────────────────────────── */
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`[DeepPlate server] running on port ${PORT}`);
  console.log(`[DeepPlate server] ML service URL: ${ML_SERVICE_URL}`);
});

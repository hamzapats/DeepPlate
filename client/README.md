# DeepPlate client

## Run locally with the inference API

1. Start the Python inference API from `models`:

   ```powershell
   python -m uvicorn api:app --host 127.0.0.1 --port 8000
   ```

2. In another terminal, start the Vite client:

   ```powershell
   npm install
   npm run dev
   ```

3. Open the Vite URL and upload a vehicle image. Vite forwards `/inference/*`
   requests to the Python API on port 8000. The sample benchmark cards remain
   illustrative; uploaded images use live model results.

The existing Express API continues to receive `/api/*` requests on port 5000.

# FORENSAI – Intelligent Acoustic Forest Protection System

## Overview
FORENSAI is a full-stack hackathon project that detects forest threats from audio, predicts risk, visualizes zones, stores incidents in MongoDB, and generates legal evidence with simulated blockchain hashing.

## Tech Stack
- Frontend: React.js, Tailwind CSS, Mapbox
- Backend: Python FastAPI
- AI: PyTorch, Librosa, scikit-learn
- Database: MongoDB
- Optional simulation: IPFS / Polygon style evidence hash generation

## Project Structure
- `frontend/` — React application
- `backend/` — FastAPI server, model, services
- `dataset/` — dataset preparation and synthetic gunshot samples
- `database/` — schema documentation

## Setup
### 1. Backend
1. Create a Python virtual environment and activate it.
2. Install dependencies:
   ```bash
   cd backend
   python -m pip install -r requirements.txt
   ```
3. Create a `.env` file if needed from `.env.sample`.
4. Ensure MongoDB is running locally or update `MONGO_URI`.
5. Prepare dataset:
   ```bash
   cd ../dataset
   python prepare_dataset.py
   ```
6. Download ESC-50 and place the folder under `dataset/ESC-50-master` with audio and `meta/esc50.csv`.
7. Add custom gunshot `.wav` files to `dataset/gunshot_samples/` or use the generated ones.
8. Train model:
   ```bash
   cd ../backend
   python model/train.py
   ```
9. Start backend server:
   ```bash
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```

> If PyTorch installation was interrupted or you are using CPU-only Windows, install with the CPU wheel:
> ```powershell
> cd backend
> python -m pip install --upgrade pip setuptools wheel
> python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchaudio
> python -m pip install -r requirements_cpu.txt
> ```

### 2. Frontend
1. Install dependencies:
   ```bash
   cd ../frontend
   npm install
   ```
2. (Optional) Create `.env` with your Mapbox token:
   ```bash
   VITE_MAPBOX_TOKEN=your_mapbox_token
   VITE_API_BASE=http://127.0.0.1:8000
   ```
   If no token is provided, the dashboard falls back to OpenStreetMap tiles automatically.
3. Start frontend:
   ```bash
   npx vite --host 127.0.0.1 --port 4173
   ```

### 3. Browser access
- Backend API: http://127.0.0.1:8000
- Frontend app: http://127.0.0.1:4173

> Do not use `0.0.0.0` in the browser address bar; it is a bind address for the server only.

## API Endpoints
- `POST /detect` — upload audio, return predicted class, confidence, risk, predicted zone, report, evidence hash
- `POST /context` — returns risk level for timestamp/location/zone
- `GET /events` — list incident events
- `GET /alerts` — list active alerts
- `GET /predictions` — list predicted zone transitions
- `GET /evidence` — list stored evidence records

## Notes
- Detection returns `Unknown` when confidence is below 0.7.
- Prediction engine follows a rule: `B2 -> C2` and similar adjacent zone mappings.
- Legal evidence generator stores JSON report and hash in MongoDB.
- Mapbox uses zone colors: green safe, yellow predicted, red danger.

# SkinVision AI - Production Ready

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# 1. Set up environment variables
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
# Edit .env files with your values

# 2. Build and start
docker compose up --build -d

# 3. Access the application
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Local Development

**Backend:**
```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## 📁 Project Structure

```
skinvision_ai/
├── backend/          # FastAPI backend
├── frontend/         # React frontend
├── model/            # PyTorch model and utilities
├── data/             # Dataset metadata
├── docker-compose.yml
└── README.md
```

## 🔐 Environment Variables

See `backend/.env.example` and `frontend/.env.example` for required variables.

**Important:** Never commit `.env` files to git!

## 🐳 Docker Services

- **backend**: FastAPI application (port 8000)
- **frontend**: React application with Nginx (port 3000)
- **db**: PostgreSQL database (port 5432)

## 📝 Model

Place your trained PyTorch model at:
```
model/efficientnet_b0_best.pth
```

The backend will automatically load it on startup.

## 🔒 Security

- Change `JWT_SECRET` in production
- Use strong database passwords
- Enable HTTPS in production
- Configure CORS properly

## 📚 Documentation

- See `DEPLOYMENT.md` for production deployment guide
- API documentation: http://localhost:8000/docs

## 🧪 Testing

```bash
cd backend
pytest
```

## 📦 Deployment

See `DEPLOYMENT.md` for detailed production deployment instructions.

# Production deployment guide
# This file contains instructions for deploying SkinVision AI to production

## Prerequisites
- Docker and Docker Compose installed
- Model file: `model/efficientnet_b0_best.pth`
- Environment variables configured

## Quick Start

1. **Set up environment variables:**
   ```bash
   cp backend/.env.example backend/.env
   cp frontend/.env.example frontend/.env
   # Edit .env files with your production values
   ```

2. **Build and start services:**
   ```bash
   docker compose up --build -d
   ```

3. **Check service status:**
   ```bash
   docker compose ps
   ```

4. **View logs:**
   ```bash
   docker compose logs -f
   ```

## Production Checklist

- [ ] Change JWT_SECRET in backend/.env
- [ ] Change database password in backend/.env
- [ ] Update FRONTEND_URL to your domain
- [ ] Update VITE_API_BASE to your backend URL
- [ ] Ensure model file exists at model/efficientnet_b0_best.pth
- [ ] Set up SSL/TLS certificates (use nginx reverse proxy)
- [ ] Configure firewall rules
- [ ] Set up database backups
- [ ] Monitor logs and health checks

## Environment Variables

### Backend (.env)
- `DATABASE_URL`: PostgreSQL connection string
- `JWT_SECRET`: Secret key for JWT tokens (CHANGE IN PRODUCTION!)
- `MODEL_PATH`: Path to PyTorch model file
- `FRONTEND_URL`: Frontend URL for CORS

### Frontend (.env)
- `VITE_API_BASE`: Backend API URL

## Health Checks

All services include health checks:
- Backend: http://localhost:8000/
- Frontend: http://localhost:3000/
- Database: PostgreSQL readiness check

## Scaling

To scale backend workers, edit `backend/Dockerfile`:
```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

## Security Notes

1. Never commit .env files to git
2. Use strong passwords for database
3. Rotate JWT_SECRET regularly
4. Use HTTPS in production
5. Set up rate limiting
6. Configure CORS properly


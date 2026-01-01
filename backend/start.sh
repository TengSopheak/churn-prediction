#!/bin/bash
cd backend
gunicorn app.main:app -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

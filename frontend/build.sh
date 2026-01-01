#!/bin/bash
cd frontend
sed -i "s|%PREDICTION_API_URL%|$PREDICTION_API_URL|g" script.js

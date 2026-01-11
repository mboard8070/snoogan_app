#!/bin/bash

# Snoogans Dashboard Stop Script
# Usage: ./stop.sh

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${RED}Stopping Snoogans Dashboard...${NC}"

pkill -f "streamlit run app.py" 2>/dev/null && echo -e "${GREEN}Streamlit (Trading) stopped${NC}" || echo "Streamlit was not running"
pkill -f "uvicorn backend.main:app" 2>/dev/null && echo -e "${GREEN}FastAPI (Backend) stopped${NC}" || echo "Backend was not running"
pkill -f "vite" 2>/dev/null && echo -e "${GREEN}React (Frontend) stopped${NC}" || echo "Frontend was not running"

echo -e "${GREEN}Done${NC}"

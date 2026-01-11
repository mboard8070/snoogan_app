#!/bin/bash

# Snoogans Dashboard Startup Script
# Usage: ./start.sh

PROJECT_DIR="/home/mboard76/nvidia-workbench/snoogan_app"
LOGS_DIR="$PROJECT_DIR/logs"

# Create logs directory if it doesn't exist
mkdir -p "$LOGS_DIR"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   SNOOGANS Dashboard Startup${NC}"
echo -e "${GREEN}========================================${NC}"

# Kill any existing processes on our ports
echo -e "${YELLOW}Cleaning up existing processes...${NC}"
pkill -f "uvicorn backend.main:app" 2>/dev/null
pkill -f "vite" 2>/dev/null
sleep 1

# Start Backend
echo -e "${YELLOW}Starting FastAPI backend on port 8000...${NC}"
cd "$PROJECT_DIR"
source venv/bin/activate
nohup uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload > "$LOGS_DIR/backend.log" 2>&1 &
BACKEND_PID=$!
echo -e "${GREEN}Backend started (PID: $BACKEND_PID)${NC}"

# Start Frontend
echo -e "${YELLOW}Starting React frontend on port 5173...${NC}"
cd "$PROJECT_DIR/frontend"
nohup npm run dev > "$LOGS_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo -e "${GREEN}Frontend started (PID: $FRONTEND_PID)${NC}"

# Wait a moment for services to start
sleep 3

# Check if services are running
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   Status${NC}"
echo -e "${GREEN}========================================${NC}"

if ps -p $BACKEND_PID > /dev/null 2>&1; then
    echo -e "${GREEN}Backend:  Running on http://localhost:8000${NC}"
else
    echo -e "${RED}Backend:  Failed to start - check $LOGS_DIR/backend.log${NC}"
fi

if ps -p $FRONTEND_PID > /dev/null 2>&1; then
    echo -e "${GREEN}Frontend: Running on http://localhost:5173${NC}"
else
    echo -e "${RED}Frontend: Failed to start - check $LOGS_DIR/frontend.log${NC}"
fi

echo ""
echo -e "${YELLOW}From your local machine, run:${NC}"
echo -e "ssh -L 5173:localhost:5173 -L 8000:localhost:8000 mboard76@100.107.132.16"
echo ""
echo -e "${YELLOW}Then open: ${GREEN}http://localhost:5173${NC}"
echo ""
echo -e "${YELLOW}To view logs:${NC}"
echo -e "  tail -f $LOGS_DIR/backend.log"
echo -e "  tail -f $LOGS_DIR/frontend.log"
echo ""
echo -e "${YELLOW}To stop services:${NC}"
echo -e "  pkill -f 'uvicorn backend.main:app'"
echo -e "  pkill -f 'vite'"
echo ""

#!/bin/bash

# Snoogans Dashboard Stop Script
# Usage: ./stop.sh

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${RED}Stopping Snoogans Dashboard...${NC}"

pkill -f "uvicorn backend.main:app" 2>/dev/null && echo -e "${GREEN}Backend stopped${NC}" || echo "Backend was not running"
pkill -f "vite" 2>/dev/null && echo -e "${GREEN}Frontend stopped${NC}" || echo "Frontend was not running"

echo -e "${GREEN}Done${NC}"

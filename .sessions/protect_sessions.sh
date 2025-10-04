#!/bin/bash
echo "🛡️ .sessions Folder Protection Check"
if [ ! -d ".sessions" ]; then mkdir -p .sessions; fi
if [ ! -f ".sessions/.gitkeep" ]; then
    echo "# .sessions directory tracker" > .sessions/.gitkeep
fi
if [ ! -f ".sessions/.gitignore" ]; then
    echo -e "# Track everything\n!*\n!*/" > .sessions/.gitignore
fi
git add -f .sessions/.gitkeep .sessions/.gitignore 2>/dev/null || true
echo "✅ .sessions protected and tracked"

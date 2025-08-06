#!/bin/bash

# Unity build script for PhosphoQuest
# This script builds the Quest APK using Unity in batch mode

# Detect OS and set Unity path accordingly
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    UNITY_PATH="/Applications/Unity/Hub/Editor/6.0.0f1/Unity.app/Contents/MacOS/Unity"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    UNITY_PATH="$HOME/Unity/Hub/Editor/6.0.0f1/Editor/Unity"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # Windows
    UNITY_PATH="C:/Program Files/Unity/Hub/Editor/6.0.0f1/Editor/Unity.exe"
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

# Check if Unity exists
if [ ! -f "$UNITY_PATH" ] && [ ! -x "$UNITY_PATH" ]; then
    echo "Unity not found at: $UNITY_PATH"
    echo "Please update the UNITY_PATH variable with your Unity installation path"
    exit 1
fi

PROJECT_PATH="$(pwd)"
BUILD_PATH="$(pwd)/Builds"
LOG_FILE="$BUILD_PATH/build.log"

# Create build directory if it doesn't exist
mkdir -p "$BUILD_PATH"

echo "========================================="
echo "Building PhosphoQuest for Meta Quest"
echo "========================================="
echo "Unity Path: $UNITY_PATH"
echo "Project Path: $PROJECT_PATH"
echo "Build Output: $BUILD_PATH/PhosphoQuest.apk"
echo "========================================="

# Run Unity build
"$UNITY_PATH" -batchmode -nographics -silent-crashes \
  -projectPath "$PROJECT_PATH" \
  -buildTarget Android \
  -executeMethod PhosphoQuest.Editor.BuildScript.BuildQuest \
  -logFile "$LOG_FILE" \
  -quit

# Check build result
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo "APK location: $BUILD_PATH/PhosphoQuest.apk"
    echo ""
    
    # Show APK size
    if [ -f "$BUILD_PATH/PhosphoQuest.apk" ]; then
        APK_SIZE=$(du -h "$BUILD_PATH/PhosphoQuest.apk" | cut -f1)
        echo "APK Size: $APK_SIZE"
    fi
else
    echo ""
    echo "❌ Build failed!"
    echo "Check the log file for details: $LOG_FILE"
    echo ""
    # Show last 20 lines of log
    tail -20 "$LOG_FILE"
    exit 1
fi
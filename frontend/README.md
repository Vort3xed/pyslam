# SLAM Viewer Frontend

A Next.js web application that streams camera frames to a WebSocket SLAM server and displays real-time pose tracking on a 2D minimap.

## Features

- **Full-screen camera view**: Camera spans the entire screen with optional mirror mode
- **Real-time minimap**: Shows movement trajectory as a line trace at top-right
- **Sensor streaming**: Sends camera frames, GPS, and compass data to SLAM server
- **Delta-based tracking**: Receives pose deltas from server to update position incrementally
- **Mobile-friendly**: Works on mobile devices with touch controls for minimap

## Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm run dev
```

3. Open http://localhost:3000 in your browser

## Usage

### Running the SLAM Server

First, start the WebSocket SLAM server from the pyslam root directory:

```bash
# Install websockets if not already installed
pip install websockets

# Run the server
python websocket_slam_server.py --port 8765 --width 640 --height 480
```

### Connecting from the Frontend

1. Open the frontend in a browser
2. Enter the WebSocket server URL (default: `ws://localhost:8765`)
3. Click "Start Camera" to begin streaming
4. Grant camera permissions when prompted
5. Move around to see the trajectory on the minimap

### Minimap Controls

- **Scroll/Pinch**: Zoom in/out
- **Drag**: Pan the view
- **+/-**: Zoom buttons
- **⌖**: Center on current position
- **↺**: Reset position tracking

## Data Flow

1. Frontend captures camera frames at ~30fps
2. Frames are sent to server as base64-encoded JPEG with optional GPS/compass data
3. Server runs SLAM on each frame (no frame skipping)
4. Server sends back pose **deltas** (changes in position/rotation)
5. Frontend accumulates deltas to maintain position state
6. Minimap visualizes the accumulated trajectory

## Notes

- GPS and compass data are collected but not yet used for error correction
- The system uses delta-based updates so relocalization errors won't reset the frontend position
- For best results, move the camera slowly and steadily


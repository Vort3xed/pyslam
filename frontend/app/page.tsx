"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import CameraStream from "./components/CameraStream";
import Minimap from "./components/Minimap";
import StatusOverlay from "./components/StatusOverlay";

// Types for pose data
interface PoseDelta {
  dx: number;
  dy: number;
  dz: number;
  dyaw: number;
  dpitch: number;
  droll: number;
  timestamp: number;
  state: string;
  frame_id: number;
}

interface Position {
  x: number;
  y: number;
  z: number;
  yaw: number;
}

interface ServerStatus {
  connected: boolean;
  queueSize: number;
  framesReceived: number;
  framesProcessed: number;
  trackingState: string;
}

// Default WebSocket URL - adjust based on your server
const DEFAULT_WS_URL = "ws://192.168.1.174:8765";

export default function Home() {
  const [isStarted, setIsStarted] = useState(false);
  const [wsUrl, setWsUrl] = useState(DEFAULT_WS_URL);
  const [error, setError] = useState<string | null>(null);
  const [serverStatus, setServerStatus] = useState<ServerStatus>({
    connected: false,
    queueSize: 0,
    framesReceived: 0,
    framesProcessed: 0,
    trackingState: "INIT",
  });

  // Position tracking - accumulates deltas
  const [position, setPosition] = useState<Position>({ x: 0, y: 0, z: 0, yaw: 0 });
  const [pathHistory, setPathHistory] = useState<Position[]>([{ x: 0, y: 0, z: 0, yaw: 0 }]);
  
  // Use refs to track actual position for path history (avoids React state sync issues)
  const positionRef = useRef<Position>({ x: 0, y: 0, z: 0, yaw: 0 });
  const lastPathPointRef = useRef<Position>({ x: 0, y: 0, z: 0, yaw: 0 });

  // WebSocket ref
  const wsRef = useRef<WebSocket | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Sensor data refs
  const gpsDataRef = useRef<{ lat: number; lon: number; accuracy: number } | null>(null);
  const compassDataRef = useRef<{ heading: number; accuracy: number } | null>(null);

  // Frame sending interval ref
  const frameSendIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Handle pose delta updates from server
  const handlePoseDeltas = useCallback((deltas: PoseDelta[]) => {
    let hasValidDelta = false;
    
    // Apply all deltas to the ref (source of truth)
    for (const delta of deltas) {
      if (delta.state === "OK") {
        positionRef.current = {
          x: positionRef.current.x + delta.dx,
          y: positionRef.current.y + delta.dy,
          z: positionRef.current.z + delta.dz,
          yaw: positionRef.current.yaw + delta.dyaw,
        };
        hasValidDelta = true;
      }
      
      // Update tracking state
      setServerStatus((prev) => ({
        ...prev,
        trackingState: delta.state,
      }));
    }
    
    // Update position state from ref
    setPosition({ ...positionRef.current });
    
    // Update path history if we moved enough
    if (hasValidDelta) {
      const lastPoint = lastPathPointRef.current;
      const currentPos = positionRef.current;
      
      const dx = currentPos.x - lastPoint.x;
      const dz = currentPos.z - lastPoint.z;
      const distance = Math.sqrt(dx * dx + dz * dz);
      
      // Add point if moved more than threshold
      if (distance > 0.01) {
        const newPoint = { ...currentPos };
        lastPathPointRef.current = newPoint;
        setPathHistory((prev) => [...prev, newPoint]);
      }
    }
  }, []);

  // Connect to WebSocket server
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log("WebSocket connected");
      setServerStatus((prev) => ({ ...prev, connected: true }));
      setError(null);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === "pose_deltas") {
          handlePoseDeltas(data.deltas);
          setServerStatus((prev) => ({
            ...prev,
            queueSize: data.queue_size,
            framesReceived: data.frames_received,
            framesProcessed: data.frames_processed,
          }));
        } else if (data.type === "pong") {
          // Heartbeat response
        }
      } catch (e) {
        console.error("Error parsing message:", e);
      }
    };

    ws.onclose = () => {
      console.log("WebSocket disconnected");
      setServerStatus((prev) => ({ ...prev, connected: false }));
    };

    ws.onerror = (e) => {
      console.error("WebSocket error:", e);
      setError("Failed to connect to SLAM server. Is it running?");
      setServerStatus((prev) => ({ ...prev, connected: false }));
    };

    wsRef.current = ws;
  }, [wsUrl, handlePoseDeltas]);

  // Start camera and sensor streams
  const startCamera = useCallback(async () => {
    try {
      // Request camera access with preference for back camera on mobile
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      // Create offscreen canvas for frame capture
      if (!canvasRef.current) {
        const canvas = document.createElement("canvas");
        canvas.width = 640;
        canvas.height = 480;
        canvasRef.current = canvas;
      }

      // Start GPS tracking
      if ("geolocation" in navigator) {
        navigator.geolocation.watchPosition(
          (position) => {
            gpsDataRef.current = {
              lat: position.coords.latitude,
              lon: position.coords.longitude,
              accuracy: position.coords.accuracy,
            };
          },
          (error) => {
            console.warn("GPS error:", error.message);
          },
          {
            enableHighAccuracy: true,
            maximumAge: 1000,
            timeout: 5000,
          }
        );
      }

      // Start compass/magnetometer tracking
      if ("DeviceOrientationEvent" in window) {
        const handleOrientation = (event: DeviceOrientationEvent) => {
          if (event.alpha !== null) {
            compassDataRef.current = {
              heading: event.alpha,
              accuracy: 0, // DeviceOrientationEvent doesn't provide accuracy
            };
          }
        };

        // Check if we need to request permission (iOS 13+)
        if (
          typeof (DeviceOrientationEvent as unknown as { requestPermission?: () => Promise<string> })
            .requestPermission === "function"
        ) {
          try {
            const permission = await (
              DeviceOrientationEvent as unknown as { requestPermission: () => Promise<string> }
            ).requestPermission();
            if (permission === "granted") {
              window.addEventListener("deviceorientation", handleOrientation);
            }
          } catch (e) {
            console.warn("Compass permission denied:", e);
          }
        } else {
          window.addEventListener("deviceorientation", handleOrientation);
        }
      }

      return true;
    } catch (e) {
      console.error("Failed to start camera:", e);
      setError("Failed to access camera. Please grant camera permissions.");
      return false;
    }
  }, []);

  // Send frame to server
  const sendFrame = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return;
    }

    if (!videoRef.current || !canvasRef.current) {
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    if (!ctx) return;

    // Draw current video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to base64 JPEG
    const frameData = canvas.toDataURL("image/jpeg", 0.8);

    // Build message with frame and sensor data
    const message: Record<string, unknown> = {
      type: "frame",
      frame: frameData,
      timestamp: Date.now() / 1000,
    };

    // Add GPS if available
    if (gpsDataRef.current) {
      message.gps_lat = gpsDataRef.current.lat;
      message.gps_lon = gpsDataRef.current.lon;
      message.gps_accuracy = gpsDataRef.current.accuracy;
    }

    // Add compass if available
    if (compassDataRef.current) {
      message.compass_heading = compassDataRef.current.heading;
      message.compass_accuracy = compassDataRef.current.accuracy;
    }

    wsRef.current.send(JSON.stringify(message));
  }, []);

  // Start streaming
  const handleStart = useCallback(async () => {
    setError(null);

    // Connect to WebSocket
    connectWebSocket();

    // Start camera
    const cameraStarted = await startCamera();
    if (!cameraStarted) {
      return;
    }

    setIsStarted(true);

    // Start sending frames at ~30fps
    // Using setInterval to ensure consistent frame rate
    frameSendIntervalRef.current = setInterval(sendFrame, 33);
  }, [connectWebSocket, startCamera, sendFrame]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (frameSendIntervalRef.current) {
        clearInterval(frameSendIntervalRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  // Reset position
  const handleReset = useCallback(() => {
    const origin = { x: 0, y: 0, z: 0, yaw: 0 };
    positionRef.current = { ...origin };
    lastPathPointRef.current = { ...origin };
    setPosition({ ...origin });
    setPathHistory([{ ...origin }]);
  }, []);

  return (
    <div className="camera-container">
      {/* Camera stream */}
      <CameraStream videoRef={videoRef} isStarted={isStarted} />

      {/* Connection prompt */}
      {!isStarted && (
        <div className="connection-prompt">
          <h2>SLAM Viewer</h2>
          <div className="mb-4">
            <label className="block text-sm mb-2">WebSocket Server URL:</label>
            <input
              type="text"
              value={wsUrl}
              onChange={(e) => setWsUrl(e.target.value)}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded text-white text-sm"
              placeholder="ws://localhost:8765"
            />
          </div>
          <button onClick={handleStart}>Start Camera</button>
          {error && <p className="text-red-400 text-sm mt-4">{error}</p>}
        </div>
      )}

      {/* Minimap overlay */}
      {isStarted && (
        <Minimap position={position} pathHistory={pathHistory} onReset={handleReset} />
      )}

      {/* Status overlay */}
      {isStarted && <StatusOverlay serverStatus={serverStatus} position={position} />}

      {/* Error overlay */}
      {isStarted && error && (
        <div className="error-message">
          <h3>Error</h3>
          <p>{error}</p>
        </div>
      )}
    </div>
  );
}

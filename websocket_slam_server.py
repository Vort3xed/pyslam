#!/usr/bin/env -S python3 -O
"""
WebSocket SLAM Server
Receives camera frames from web clients via WebSocket, processes them through SLAM,
and sends back camera pose deltas for visualization.

Usage:
    python websocket_slam_server.py --port 8765 --width 640 --height 480
"""

import cv2
import time
import os
import sys
import numpy as np
import json
import threading
import asyncio
import base64
from queue import Queue, Empty
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Set

from pyslam.config import Config
from pyslam.slam.slam import Slam, SlamState
from pyslam.slam.camera import PinholeCamera
from pyslam.io.dataset_types import DatasetType, SensorType
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
from pyslam.loop_closing.loop_detector_configs import LoopDetectorConfigs
from pyslam.utilities.utils_sys import Printer
from pyslam.utilities.utils_serialization import SerializableEnumEncoder
from pyslam.viz.viewer3D import Viewer3D
from pyslam.utilities.utils_img import ImgWriter

import argparse
from datetime import datetime

# Import websockets
try:
    import websockets
    from websockets.server import serve
except ImportError:
    print("Please install websockets: pip install websockets")
    sys.exit(1)


@dataclass
class SensorData:
    """Container for sensor data from client"""
    frame: Optional[np.ndarray] = None
    timestamp: float = 0.0
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    gps_accuracy: Optional[float] = None
    compass_heading: Optional[float] = None
    compass_accuracy: Optional[float] = None


@dataclass
class PoseDelta:
    """Camera pose delta to send to client"""
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    dyaw: float = 0.0  # Rotation around Y axis (heading change)
    dpitch: float = 0.0  # Rotation around X axis
    droll: float = 0.0  # Rotation around Z axis
    timestamp: float = 0.0
    state: str = "INIT"
    frame_id: int = 0


class WebSocketSLAMDataset:
    """Dataset wrapper for WebSocket frame input - processes ALL frames in order"""
    
    def __init__(self, frame_queue: Queue, width=640, height=480):
        self.frame_queue = frame_queue
        self.target_width = width
        self.target_height = height
        self.is_ok = True
        self.sensor_type = SensorType.MONOCULAR
        self.img_id = 0
        self.type = DatasetType.VIDEO
        self.scale_viewer_3d = 1.0
        self.start_time = time.time()
        self.last_frame_time = None
        self.frame_interval = 1.0 / 30.0
        self._current_sensor_data: Optional[SensorData] = None
        
    def getImageColor(self, img_id):
        """Get next frame from queue - blocks until frame available"""
        try:
            # Wait for next frame with timeout
            sensor_data: SensorData = self.frame_queue.get(timeout=5.0)
            self._current_sensor_data = sensor_data
            
            frame = sensor_data.frame
            if frame is None:
                return None
                
            # Resize to expected camera dimensions
            if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
                frame = cv2.resize(frame, (self.target_width, self.target_height))
            
            self.last_frame_time = sensor_data.timestamp
            self.img_id = img_id
            return frame
            
        except Empty:
            return None
    
    def getCurrentSensorData(self) -> Optional[SensorData]:
        """Get the sensor data for the current frame (GPS, compass)"""
        return self._current_sensor_data
    
    def getDepth(self, img_id):
        return None
    
    def getImageColorRight(self, img_id):
        return None
    
    def getTimestamp(self):
        if self.last_frame_time is not None:
            return self.last_frame_time
        return time.time() - self.start_time
    
    def getNextTimestamp(self):
        return self.getTimestamp() + self.frame_interval
    
    def environmentType(self):
        return None
    
    def sensorType(self):
        return SensorType.MONOCULAR


class SLAMProcessor:
    """Handles SLAM processing in a separate thread"""
    
    def __init__(self, config: Config, width=640, height=480, headless=True, debug=False, skip_frames=0):
        self.config = config
        self.width = width
        self.height = height
        self.headless = headless
        self.debug = debug  # Debug mode enables 3D viewer and camera display
        self.skip_frames = skip_frames  # Process every (skip_frames+1)th frame
        
        # Frame queue - unlimited size to ensure no frames are dropped
        self.frame_queue: Queue[SensorData] = Queue(maxsize=0)
        
        # Result queue for pose deltas
        self.result_queue: Queue[PoseDelta] = Queue(maxsize=100)
        
        # Previous pose for delta calculation
        self.prev_position: Optional[np.ndarray] = None
        self.prev_rotation: Optional[np.ndarray] = None
        
        # State
        self.running = False
        self.slam = None
        self.processing_thread = None
        self.frames_processed = 0
        self.frames_received = 0
        self.frames_skipped = 0  # Counter for frame skipping
        self.last_state = "INIT"
        
        # Debug visualization objects
        self.viewer3D = None
        self.img_writer = None
        
    def start(self):
        """Start the SLAM processing thread"""
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        Printer.green("SLAM processor started")
        
    def stop(self):
        """Stop the SLAM processing thread"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        if self.slam:
            self.slam.quit()
        if self.viewer3D:
            self.viewer3D.quit()
        if self.debug:
            cv2.destroyAllWindows()
        Printer.yellow("SLAM processor stopped")
    
    def add_frame(self, sensor_data: SensorData):
        """Add a frame to the processing queue (with optional frame skipping)"""
        self.frames_received += 1
        
        # Skip frames at the receiving end to maintain temporal consistency
        # This ensures skipped frames never enter the queue, so timestamps remain smooth
        if self.skip_frames > 0:
            if self.frames_received % (self.skip_frames + 1) != 1:
                self.frames_skipped += 1
                return  # Don't add this frame to queue
        
        self.frame_queue.put(sensor_data)
        
    def get_pose_delta(self) -> Optional[PoseDelta]:
        """Get the next pose delta if available"""
        try:
            return self.result_queue.get_nowait()
        except Empty:
            return None
    
    def get_all_pose_deltas(self) -> list:
        """Get all available pose deltas"""
        deltas = []
        while True:
            try:
                deltas.append(self.result_queue.get_nowait())
            except Empty:
                break
        return deltas
    
    def _rotation_matrix_to_euler(self, R: np.ndarray) -> tuple:
        """Convert rotation matrix to euler angles (roll, pitch, yaw)"""
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
            
        return roll, pitch, yaw
    
    def _calculate_pose_delta(self, cur_R: np.ndarray, cur_t: np.ndarray, 
                              timestamp: float, frame_id: int, state: str) -> PoseDelta:
        """Calculate the delta between current and previous pose"""
        cur_pos = cur_t.flatten()
        
        if self.prev_position is None or self.prev_rotation is None:
            # First valid pose - no delta yet
            self.prev_position = cur_pos.copy()
            self.prev_rotation = cur_R.copy()
            return PoseDelta(
                dx=0.0, dy=0.0, dz=0.0,
                dyaw=0.0, dpitch=0.0, droll=0.0,
                timestamp=timestamp,
                state=state,
                frame_id=frame_id
            )
        
        # Calculate position delta
        dx = float(cur_pos[0] - self.prev_position[0])
        dy = float(cur_pos[1] - self.prev_position[1])
        dz = float(cur_pos[2] - self.prev_position[2])
        
        # Calculate rotation delta
        prev_roll, prev_pitch, prev_yaw = self._rotation_matrix_to_euler(self.prev_rotation)
        cur_roll, cur_pitch, cur_yaw = self._rotation_matrix_to_euler(cur_R)
        
        droll = float(cur_roll - prev_roll)
        dpitch = float(cur_pitch - prev_pitch)
        dyaw = float(cur_yaw - prev_yaw)
        
        # Normalize angle deltas to [-pi, pi]
        dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
        dpitch = (dpitch + np.pi) % (2 * np.pi) - np.pi
        droll = (droll + np.pi) % (2 * np.pi) - np.pi
        
        # Update previous pose
        self.prev_position = cur_pos.copy()
        self.prev_rotation = cur_R.copy()
        
        return PoseDelta(
            dx=dx, dy=dy, dz=dz,
            dyaw=dyaw, dpitch=dpitch, droll=droll,
            timestamp=timestamp,
            state=state,
            frame_id=frame_id
        )
    
    def _processing_loop(self):
        """Main SLAM processing loop"""
        # Initialize camera
        camera = PinholeCamera(self.config)
        
        # Create dataset wrapper
        dataset = WebSocketSLAMDataset(self.frame_queue, self.width, self.height)
        dataset.target_width = camera.width
        dataset.target_height = camera.height
        
        # Feature tracker configuration - ORB2 for speed
        feature_tracker_config = FeatureTrackerConfigs.ORB2
        
        # Loop detection
        loop_detection_config = LoopDetectorConfigs.DBOW3
        
        # No semantic mapping for WebSocket streaming
        semantic_mapping_config = None
        
        self.config.feature_tracker_config = feature_tracker_config
        self.config.loop_detection_config = loop_detection_config
        self.config.semantic_mapping_config = semantic_mapping_config
        
        Printer.green(
            "feature_tracker_config: ",
            json.dumps(feature_tracker_config, indent=4, cls=SerializableEnumEncoder),
        )
        
        # Create SLAM
        self.slam = Slam(
            camera,
            feature_tracker_config,
            loop_detection_config,
            semantic_mapping_config,
            dataset.sensorType(),
            environment_type=dataset.environmentType(),
            config=self.config,
            headless=self.headless,
        )
        
        Printer.green(f"SLAM initialized: {camera.width}x{camera.height}")
        
        # Initialize debug visualization if enabled
        if self.debug:
            Printer.green("Debug mode enabled - starting 3D viewer and camera display")
            self.viewer3D = Viewer3D(scale=dataset.scale_viewer_3d)
            self.img_writer = ImgWriter(font_scale=0.7)
            time.sleep(1)  # Give viewer time to initialize
        
        img_id = 0
        consecutive_lost = 0
        max_lost_before_reset = 30
        last_good_position = None
        last_good_rotation = None
        fps_counter = time.time()
        fps_display = 0.0
        frames_since_fps = 0
        
        while self.running:
            img = dataset.getImageColor(img_id)
            
            if img is None:
                continue
            
            # Note: Frame skipping is done at the receiving end (add_frame method)
            # to maintain temporal consistency for the motion model
                
            timestamp = dataset.getTimestamp()
            
            # Save current good position before tracking
            if self.slam.tracking.cur_R is not None and self.slam.tracking.cur_t is not None:
                if self.slam.tracking.state == SlamState.OK:
                    last_good_position = self.slam.tracking.cur_t.copy()
                    last_good_rotation = self.slam.tracking.cur_R.copy()
            
            # Main SLAM tracking
            try:
                self.slam.track(img, None, None, img_id, timestamp)
            except AssertionError as e:
                Printer.red(f"Tracking assertion error: {e}")
                try:
                    self.slam.reset()
                except:
                    pass
                self.prev_position = None
                self.prev_rotation = None
                consecutive_lost = 0
                img_id += 1
                continue
            except Exception as e:
                Printer.red(f"Tracking error: {e}")
                img_id += 1
                continue
            
            self.frames_processed += 1
            
            # Track consecutive lost frames
            current_state = self.slam.tracking.state.name
            if self.slam.tracking.state == SlamState.LOST or self.slam.tracking.state == SlamState.RELOCALIZE:
                consecutive_lost += 1
            else:
                consecutive_lost = 0
            
            # Auto-reset if stuck in LOST state
            if consecutive_lost >= max_lost_before_reset:
                Printer.yellow(f"Lost for {consecutive_lost} frames, resetting...")
                try:
                    self.slam.reset()
                except:
                    pass
                self.prev_position = None
                self.prev_rotation = None
                consecutive_lost = 0
            
            # Calculate FPS for debug display
            frames_since_fps += 1
            if time.time() - fps_counter > 1.0:
                fps_display = frames_since_fps / (time.time() - fps_counter)
                frames_since_fps = 0
                fps_counter = time.time()
            
            # Calculate and queue pose delta
            if self.slam.tracking.cur_R is not None and self.slam.tracking.cur_t is not None:
                pose_delta = self._calculate_pose_delta(
                    self.slam.tracking.cur_R,
                    self.slam.tracking.cur_t,
                    timestamp,
                    img_id,
                    current_state
                )
                
                # Add to result queue (drop oldest if full)
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except:
                        pass
                self.result_queue.put(pose_delta)
                
                # Log every 30 frames
                if img_id % 30 == 0:
                    pos = self.slam.tracking.cur_t.flatten()
                    queue_size = self.frame_queue.qsize()
                    Printer.cyan(f"Frame {img_id} | Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | State: {current_state} | Queue: {queue_size}")
            else:
                # Send a "no pose" delta
                pose_delta = PoseDelta(
                    dx=0.0, dy=0.0, dz=0.0,
                    dyaw=0.0, dpitch=0.0, droll=0.0,
                    timestamp=timestamp,
                    state=current_state,
                    frame_id=img_id
                )
                self.result_queue.put(pose_delta)
            
            # Debug visualization
            if self.debug:
                # Draw 3D map
                if self.viewer3D:
                    self.viewer3D.draw_slam_map(self.slam)
                    self.viewer3D.draw_dense_map(self.slam)
                    
                    # Check if viewer was closed
                    if self.viewer3D.is_closed():
                        Printer.yellow("3D viewer closed by user")
                        self.running = False
                        break
                
                # Draw 2D camera view with feature trails (every 3 frames for performance)
                if img_id % 3 == 0:
                    img_draw = self.slam.map.draw_feature_trails(img)
                    lost_info = f" (lost: {consecutive_lost})" if consecutive_lost > 0 else ""
                    self.img_writer.write(img_draw, f"Frame: {img_id} | FPS: {fps_display:.1f} | State: {current_state}{lost_info}", (30, 30))
                    cv2.imshow("WebSocket SLAM - Camera View", img_draw)
                    
                    # Handle keyboard input
                    key_cv = cv2.waitKey(1) & 0xFF
                    if key_cv == ord('q') or key_cv == 27:  # 'q' or ESC
                        Printer.yellow("Quit requested via keyboard")
                        self.running = False
                        break
            
            img_id += 1


class WebSocketServer:
    """WebSocket server that handles client connections and routes data to SLAM"""
    
    def __init__(self, slam_processor: SLAMProcessor, host="0.0.0.0", port=8765):
        self.slam_processor = slam_processor
        self.host = host
        self.port = port
        self.clients: Set = set()
        self.running = False
        
    async def handle_client(self, websocket):
        """Handle a single client connection"""
        client_id = id(websocket)
        self.clients.add(websocket)
        Printer.green(f"Client connected: {client_id} (total: {len(self.clients)})")
        
        try:
            # Start a task to send pose deltas back to this client
            send_task = asyncio.create_task(self._send_pose_deltas(websocket))
            
            async for message in websocket:
                await self._process_message(message, websocket)
                
        except websockets.exceptions.ConnectionClosed:
            Printer.yellow(f"Client disconnected: {client_id}")
        except Exception as e:
            Printer.red(f"Client error: {e}")
        finally:
            self.clients.discard(websocket)
            send_task.cancel()
            Printer.yellow(f"Client removed: {client_id} (remaining: {len(self.clients)})")
    
    async def _process_message(self, message: str, websocket):
        """Process incoming message from client"""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")
            
            if msg_type == "frame":
                # Decode base64 image
                frame_data = data.get("frame", "")
                if frame_data:
                    # Remove data URL prefix if present
                    if "," in frame_data:
                        frame_data = frame_data.split(",")[1]
                    
                    # Decode base64 to numpy array
                    img_bytes = base64.b64decode(frame_data)
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Create sensor data with optional GPS/compass
                        sensor_data = SensorData(
                            frame=frame,
                            timestamp=data.get("timestamp", time.time()),
                            gps_lat=data.get("gps_lat"),
                            gps_lon=data.get("gps_lon"),
                            gps_accuracy=data.get("gps_accuracy"),
                            compass_heading=data.get("compass_heading"),
                            compass_accuracy=data.get("compass_accuracy")
                        )
                        
                        # Add to SLAM processing queue
                        self.slam_processor.add_frame(sensor_data)
                        
            elif msg_type == "ping":
                await websocket.send(json.dumps({"type": "pong", "timestamp": time.time()}))
                
        except json.JSONDecodeError:
            Printer.red("Invalid JSON received")
        except Exception as e:
            Printer.red(f"Error processing message: {e}")
    
    async def _send_pose_deltas(self, websocket):
        """Continuously send pose deltas to client"""
        while True:
            try:
                # Get all available pose deltas
                deltas = self.slam_processor.get_all_pose_deltas()
                
                if deltas:
                    # Send all deltas as a batch
                    message = {
                        "type": "pose_deltas",
                        "deltas": [asdict(d) for d in deltas],
                        "queue_size": self.slam_processor.frame_queue.qsize(),
                        "frames_received": self.slam_processor.frames_received,
                        "frames_processed": self.slam_processor.frames_processed,
                        "frames_skipped": self.slam_processor.frames_skipped
                    }
                    await websocket.send(json.dumps(message))
                
                # Small delay to batch updates
                await asyncio.sleep(0.033)  # ~30fps update rate
                
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                Printer.red(f"Error sending pose delta: {e}")
                await asyncio.sleep(0.1)
    
    async def start(self):
        """Start the WebSocket server"""
        self.running = True
        Printer.green(f"WebSocket server starting on ws://{self.host}:{self.port}")
        
        async with serve(self.handle_client, self.host, self.port):
            Printer.green(f"WebSocket server running on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever


def main():
    parser = argparse.ArgumentParser(description="WebSocket SLAM Server")
    parser.add_argument(
        "-c", "--config_path",
        type=str,
        default=None,
        help="Optional path for custom configuration file",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to listen on (default: 8765)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Expected frame width (default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Expected frame height (default: 480)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run SLAM in headless mode (default: True)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with 3D viewer and camera visualization",
    )
    parser.add_argument(
        "--skip_frames",
        type=int,
        default=0,
        help="Process every Nth frame (0=all frames, 1=every 2nd, 2=every 3rd, etc.)",
    )
    args = parser.parse_args()
    
    # Debug mode implies not headless
    if args.debug:
        args.headless = False
    
    # Load config
    if args.config_path:
        config = Config(args.config_path)
    else:
        config = Config()
    
    print("\n" + "=" * 60)
    print("WebSocket SLAM Server")
    print("=" * 60)
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Frame size: {args.width}x{args.height}")
    print(f"  Headless: {args.headless}")
    print(f"  Debug: {args.debug}")
    print(f"  Skip frames: {args.skip_frames} (processing every {args.skip_frames + 1} frames)")
    if args.debug:
        print("  Debug controls: Press 'q' or ESC in camera window to quit")
    print("=" * 60 + "\n")
    
    # Create SLAM processor
    slam_processor = SLAMProcessor(
        config=config,
        width=args.width,
        height=args.height,
        headless=args.headless,
        debug=args.debug,
        skip_frames=args.skip_frames
    )
    
    # Start SLAM processing thread
    slam_processor.start()
    
    # Create and start WebSocket server
    server = WebSocketServer(slam_processor, args.host, args.port)
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        slam_processor.stop()
        print("Server stopped.")


if __name__ == "__main__":
    main()

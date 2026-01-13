#!/usr/bin/env -S python3 -O
"""
Modified PYSLAM script for webcam input with simplified output
Shows 3D map and prints camera coordinates in real-time
"""

import cv2
import time
import os
import sys
import numpy as np
import json

from pyslam.config import Config

from pyslam.semantics.semantic_mapping_configs import SemanticMappingConfigs
from pyslam.semantics.semantic_eval import evaluate_semantic_mapping

from pyslam.slam.slam import Slam, SlamState
from pyslam.slam.camera import PinholeCamera
from pyslam.io.dataset_types import DatasetType, SensorType
from pyslam.io.trajectory_writer import TrajectoryWriter

from pyslam.viz.viewer3D import Viewer3D
from pyslam.utilities.utils_sys import Printer
from pyslam.utilities.utils_img import ImgWriter
from pyslam.utilities.utils_serialization import SerializableEnumEncoder

from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
from pyslam.loop_closing.loop_detector_configs import LoopDetectorConfigs

from pyslam.depth_estimation.depth_estimator_factory import (
    depth_estimator_factory,
    DepthEstimatorType,
)
from pyslam.utilities.utils_depth import img_from_depth, filter_shadow_points
from pyslam.config_parameters import Parameters

from datetime import datetime
import argparse


# Webcam Dataset Wrapper
class WebcamDataset:
    def __init__(self, ip_camera_url=None, video_path=None, camera_id=0, width=640, height=480):
        self.use_manual_fetch = False
        self.ip_camera_base_url = None
        
        if ip_camera_url:
            # Use IP camera (e.g., phone with IP Webcam app)
            camera_url = ip_camera_url.rstrip('/') + "/video"
            self.cap = cv2.VideoCapture(camera_url)
            print(f"Connecting to IP camera: {camera_url}")
            
            # If OpenCV can't open it, try manual fetching
            if not self.cap.isOpened():
                print("OpenCV failed to open stream, trying manual fetch method...")
                self.use_manual_fetch = True
                self.ip_camera_base_url = ip_camera_url.rstrip('/')
                import requests
                self.session = requests.Session()
                # Test connection
                try:
                    response = self.session.get(f"{self.ip_camera_base_url}/shot.jpg", timeout=5)
                    if response.status_code != 200:
                        raise RuntimeError(f"Cannot connect to IP camera: {response.status_code}")
                    print("Manual fetch method working!")
                except Exception as e:
                    raise RuntimeError(f"Failed to connect to IP camera: {e}")
        elif video_path:
            # Use video file
            self.cap = cv2.VideoCapture(video_path)
            print(f"Loading video file: {video_path}")
        else:
            # Use local webcam
            self.cap = cv2.VideoCapture(camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            print(f"Opening local webcam: {camera_id}")
        
        self.is_ok = True
        self.sensor_type = SensorType.MONOCULAR
        self.img_id = 0
        self.type = DatasetType.VIDEO
        self.scale_viewer_3d = 1.0
        self.start_time = time.time()
        self.num_frames = 0
        self.is_video_file = video_path is not None
        
        if not self.use_manual_fetch and not self.cap.isOpened():
            raise RuntimeError("Failed to open camera/video")
        
        if not self.use_manual_fetch:
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Camera initialized: {actual_width}x{actual_height}")
        else:
            print(f"Camera initialized with manual fetch")
        
    def getImageColor(self, img_id):
        if self.use_manual_fetch:
            # Manual fetch from IP camera using requests
            try:
                import requests
                response = self.session.get(f"{self.ip_camera_base_url}/shot.jpg", timeout=2)
                if response.status_code == 200:
                    img_array = np.frombuffer(response.content, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if frame is not None:
                        # Resize to expected camera dimensions
                        if hasattr(self, 'target_width') and hasattr(self, 'target_height'):
                            frame = cv2.resize(frame, (self.target_width, self.target_height))
                        self.img_id = img_id
                        self.num_frames += 1
                        return frame
            except Exception as e:
                print(f"Error fetching frame: {e}")
                self.is_ok = False
                return None
        else:
            # Normal OpenCV capture
            ret, frame = self.cap.read()
            if ret:
                # Resize to expected camera dimensions
                if hasattr(self, 'target_width') and hasattr(self, 'target_height'):
                    frame = cv2.resize(frame, (self.target_width, self.target_height))
                self.img_id = img_id
                self.num_frames += 1
                return frame
        
        self.is_ok = False
        return None
    
    def getDepth(self, img_id):
        return None
    
    def getImageColorRight(self, img_id):
        return None
    
    def getTimestamp(self):
        return time.time() - self.start_time
    
    def getNextTimestamp(self):
        return self.getTimestamp() + 0.033  # ~30fps
    
    def environmentType(self):
        return None
    
    def sensorType(self):
        return SensorType.MONOCULAR
    
    def release(self):
        if not self.use_manual_fetch:
            self.cap.release()
        else:
            self.session.close()


datetime_string = datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default=None,
        help="Optional path for custom configuration file",
    )
    parser.add_argument(
        "--camera_id",
        type=int,
        default=0,
        help="Camera device ID (default: 0)",
    )
    parser.add_argument(
        "--ip_camera",
        type=str,
        default=None,
        help="IP camera URL (e.g., http://192.168.1.100:8080)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file for testing",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera width (default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera height (default: 480)",
    )
    parser.add_argument(
        "--no_output_date",
        action="store_true",
        help="Do not append date to output directory",
    )
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    args = parser.parse_args()

    if args.config_path:
        config = Config(args.config_path)
    else:
        config = Config()

    if args.no_output_date:
        datetime_string = None

    # Create webcam dataset
    dataset = WebcamDataset(
        ip_camera_url=args.ip_camera,
        video_path=args.video,
        camera_id=args.camera_id,
        width=args.width,
        height=args.height
    )
    is_monocular = True
    num_total_frames = float('inf')

    online_trajectory_writer = None
    final_trajectory_writer = None
    if config.trajectory_saving_settings["save_trajectory"]:
        (
            trajectory_online_file_path,
            trajectory_final_file_path,
            trajectory_saving_base_path,
        ) = config.get_trajectory_saving_paths(datetime_string)
        online_trajectory_writer = TrajectoryWriter(
            format_type=config.trajectory_saving_settings["format_type"],
            filename=trajectory_online_file_path,
        )
        final_trajectory_writer = TrajectoryWriter(
            format_type=config.trajectory_saving_settings["format_type"],
            filename=trajectory_final_file_path,
        )

    camera = PinholeCamera(config)

    # Set target resolution for webcam dataset to match camera config
    dataset.target_width = camera.width
    dataset.target_height = camera.height
    print(f"Camera config expects: {camera.width}x{camera.height}")
    print(f"Frames will be resized to match camera config")

    # Feature tracker configuration
    feature_tracker_config = FeatureTrackerConfigs.LIGHTGLUE

    # Loop detection configuration (set to None to disable)
    loop_detection_config = LoopDetectorConfigs.NETVLAD

    # Semantic mapping disabled for webcam
    semantic_mapping_config = None

    # Override configurations from settings file if provided
    if config.feature_tracker_config_name is not None:
        feature_tracker_config = FeatureTrackerConfigs.get_config_from_name(
            config.feature_tracker_config_name
        )
    if config.num_features_to_extract > 0:
        Printer.yellow(
            "Setting feature_tracker_config num_features from settings: ",
            config.num_features_to_extract,
        )
        feature_tracker_config["num_features"] = config.num_features_to_extract
    if config.loop_detection_config_name is not None:
        loop_detection_config = LoopDetectorConfigs.get_config_from_name(
            config.loop_detection_config_name
        )

    Printer.green(
        "feature_tracker_config: ",
        json.dumps(feature_tracker_config, indent=4, cls=SerializableEnumEncoder),
    )
    Printer.green(
        "loop_detection_config: ",
        json.dumps(loop_detection_config, indent=4, cls=SerializableEnumEncoder),
    )

    config.feature_tracker_config = feature_tracker_config
    config.loop_detection_config = loop_detection_config
    config.semantic_mapping_config = semantic_mapping_config

    # Depth estimator (optional)
    depth_estimator = None
    if Parameters.kUseDepthEstimatorInFrontEnd:
        Parameters.kVolumetricIntegrationUseDepthEstimator = False
        depth_estimator_type = DepthEstimatorType.DEPTH_PRO
        max_depth = 20
        depth_estimator = depth_estimator_factory(
            depth_estimator_type=depth_estimator_type,
            max_depth=max_depth,
            dataset_env_type=dataset.environmentType(),
            camera=camera,
        )
        Printer.green(f"Depth_estimator_type: {depth_estimator_type.name}, max_depth: {max_depth}")

    # Create SLAM object
    slam = Slam(
        camera,
        feature_tracker_config,
        loop_detection_config,
        semantic_mapping_config,
        dataset.sensorType(),
        environment_type=dataset.environmentType(),
        config=config,
        headless=args.headless,
    )
    slam.set_viewer_scale(dataset.scale_viewer_3d)
    time.sleep(1)

    if args.headless:
        viewer3D = None
    else:
        viewer3D = Viewer3D(scale=dataset.scale_viewer_3d)
        img_writer = ImgWriter(font_scale=0.7)

    do_step = False
    do_reset = False
    is_paused = False
    is_map_save = False
    is_viewer_closed = False

    key_cv = None
    num_frames = 0
    img_id = 0

    print("\n" + "="*60)
    print("SLAM Started - Press 'q' or ESC to quit")
    print("="*60 + "\n")

    try:
        while not is_viewer_closed and dataset.is_ok:

            if do_reset:
                Printer.yellow("Resetting SLAM...")
                slam.reset()
                do_reset = False

            if not is_paused or do_step:
                img = dataset.getImageColor(img_id)
                depth = dataset.getDepth(img_id)
                img_right = dataset.getImageColorRight(img_id)

                if img is not None:
                    timestamp = dataset.getTimestamp()
                    next_timestamp = dataset.getNextTimestamp()
                    frame_duration = next_timestamp - timestamp if timestamp is not None else -1

                    time_start = time.time()

                    # Depth estimation if enabled
                    if depth is None and depth_estimator:
                        depth_prediction, pts3d_prediction = depth_estimator.infer(img, img_right)
                        if Parameters.kDepthEstimatorRemoveShadowPointsInFrontEnd:
                            depth = filter_shadow_points(depth_prediction)
                        else:
                            depth = depth_prediction

                        if not args.headless:
                            depth_img = img_from_depth(depth_prediction, img_min=0, img_max=50)
                            cv2.imshow("depth prediction", depth_img)

                    # Main SLAM tracking
                    slam.track(img, img_right, depth, img_id, timestamp)

                    # Print camera coordinates
                    if slam.tracking.cur_R is not None and slam.tracking.cur_t is not None:
                        pos = slam.tracking.cur_t.flatten()
                        print(f"Frame {img_id:5d} | Position: ({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}) | State: {slam.tracking.state.name}")
                    else:
                        print(f"Frame {img_id:5d} | Position: LOST/INIT | State: {slam.tracking.state.name}")

                    # 3D display
                    if viewer3D:
                        viewer3D.draw_slam_map(slam)
                        viewer3D.draw_dense_map(slam)

                    # 2D display
                    if not args.headless:
                        img_draw = slam.map.draw_feature_trails(img)
                        img_writer.write(img_draw, f"Frame: {img_id} | FPS: {1.0/(time.time()-time_start):.1f}", (30, 30))
                        cv2.imshow("Camera", img_draw)

                    # Save trajectory
                    if (
                        online_trajectory_writer is not None
                        and slam.tracking.cur_R is not None
                        and slam.tracking.cur_t is not None
                    ):
                        online_trajectory_writer.write_trajectory(
                            slam.tracking.cur_R, slam.tracking.cur_t, timestamp
                        )

                    # Frame timing
                    processing_duration = time.time() - time_start
                    if frame_duration > processing_duration:
                        time.sleep(frame_duration - processing_duration)

                    img_id += 1
                    num_frames += 1
                    do_step = False
                else:
                    time.sleep(0.1)
                    if args.headless:
                        break
            else:
                time.sleep(0.1)

            # Handle keyboard input
            if not args.headless:
                if slam.tracking.state == SlamState.LOST:
                    key_cv = cv2.waitKey(500) & 0xFF
                else:
                    key_cv = cv2.waitKey(1) & 0xFF

            # Handle map save
            if is_map_save:
                slam.save_system_state(config.system_state_folder_path)
                Printer.blue("\nMap saved! Uncheck pause to continue...\n")
                is_map_save = False

            # Update viewer state
            if viewer3D:
                is_paused = viewer3D.is_paused()
                is_map_save = viewer3D.is_map_save() and is_map_save == False
                do_step = viewer3D.do_step() and do_step == False
                do_reset = viewer3D.reset() and do_reset == False
                is_viewer_closed = viewer3D.is_closed()

            # Quit on 'q' or ESC
            if key_cv == ord('q') or key_cv == 27:
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\n" + "="*60)
        print("Cleaning up...")
        print("="*60)
        
        if online_trajectory_writer:
            online_trajectory_writer.close_file()
        
        try:
            est_poses, timestamps, ids = slam.get_final_trajectory()
            if final_trajectory_writer:
                final_trajectory_writer.write_full_trajectory(est_poses, timestamps)
                final_trajectory_writer.close_file()
        except Exception as e:
            print(f"Could not save final trajectory: {e}")

        slam.quit()
        if viewer3D:
            viewer3D.quit()
        
        dataset.release()
        
        if not args.headless:
            cv2.destroyAllWindows()
        
        print("Done!")

"use client";

import { RefObject } from "react";

interface CameraStreamProps {
  videoRef: RefObject<HTMLVideoElement | null>;
  isStarted: boolean;
}

export default function CameraStream({ videoRef, isStarted }: CameraStreamProps) {
  return (
    <>
      <video
        ref={videoRef}
        className="camera-video"
        autoPlay
        playsInline
        muted
        style={{
          display: isStarted ? "block" : "none",
        }}
      />
      {!isStarted && (
        <div className="absolute inset-0 bg-black flex items-center justify-center">
          <div className="text-gray-500 text-lg">Camera preview will appear here</div>
        </div>
      )}
    </>
  );
}

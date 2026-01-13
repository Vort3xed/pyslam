"use client";

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

interface StatusOverlayProps {
  serverStatus: ServerStatus;
  position: Position;
}

export default function StatusOverlay({ serverStatus, position }: StatusOverlayProps) {
  const getStateColor = (state: string) => {
    switch (state) {
      case "OK":
        return "status-tracking";
      case "LOST":
      case "RELOCALIZE":
        return "status-lost";
      default:
        return "text-gray-400";
    }
  };

  return (
    <div className="status-overlay">
      <div className="flex flex-col gap-1">
        {/* Connection status */}
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              serverStatus.connected ? "bg-green-400" : "bg-red-400"
            }`}
          />
          <span className={serverStatus.connected ? "status-connected" : "status-disconnected"}>
            {serverStatus.connected ? "Connected" : "Disconnected"}
          </span>
        </div>

        {/* Tracking state */}
        <div>
          State:{" "}
          <span className={getStateColor(serverStatus.trackingState)}>
            {serverStatus.trackingState}
          </span>
        </div>

        {/* Queue info */}
        <div className="text-gray-400">
          Queue: {serverStatus.queueSize} | Processed: {serverStatus.framesProcessed}
        </div>

        {/* Position */}
        <div className="text-gray-400 border-t border-gray-700 pt-1 mt-1">
          X: {position.x.toFixed(2)} | Y: {position.y.toFixed(2)} | Z: {position.z.toFixed(2)}
        </div>
        <div className="text-gray-400">
          Yaw: {((position.yaw * 180) / Math.PI).toFixed(1)}°
        </div>
      </div>
    </div>
  );
}

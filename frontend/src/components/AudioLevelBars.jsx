const BAR_WEIGHTS = [0.55, 0.8, 1, 0.8, 0.55];
const BAR_DELAYS_MS = [0, 40, 80, 40, 0];
const MIN_HEIGHT_PX = 6;
const MAX_HEIGHT_PX = 28;

export default function AudioLevelBars({ level }) {
  const clamped = Math.max(0, Math.min(1, level || 0));

  return (
    <div className="flex items-end gap-1.5 h-7" aria-hidden="true">
      {BAR_WEIGHTS.map((weight, i) => {
        const height = MIN_HEIGHT_PX + clamped * weight * (MAX_HEIGHT_PX - MIN_HEIGHT_PX);
        const opacity = 0.35 + clamped * 0.65;
        return (
          <div
            key={i}
            className="w-1.5 rounded-full bg-purple-400"
            style={{
              height: `${height}px`,
              opacity,
              transition: `height 150ms ease-out ${BAR_DELAYS_MS[i]}ms, opacity 150ms ease-out ${BAR_DELAYS_MS[i]}ms`,
            }}
          />
        );
      })}
    </div>
  );
}

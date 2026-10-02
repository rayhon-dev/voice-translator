const BAR_WEIGHTS = [0.35, 0.6, 0.85, 1, 0.85, 0.6, 0.35];
const BAR_DELAYS_MS = [0, 30, 60, 90, 60, 30, 0];
const MIN_HEIGHT_PX = 8;
const MAX_HEIGHT_PX = 56;

// Energiya PAST bo'lganda xira (deyarli rangsiz) binafsha, BALAND bo'lganda
// to'liq yorqin binafshaga o'tadi — foydalanuvchi bir qarashda "eshitilyapti"
// yoki "jim turibman"ligini, matn o'qimasdan ham, darhol sezishi uchun.
const MUTED_RGB = [205, 194, 236];
const VIVID_RGB = [147, 51, 234];

function lerpColor(from, to, t) {
  return `rgb(${from.map((c, i) => Math.round(c + (to[i] - c) * t)).join(", ")})`;
}

export default function AudioLevelBars({ level }) {
  const clamped = Math.max(0, Math.min(1, level || 0));
  const isActive = clamped > 0.04;

  return (
    <div
      className="flex items-end gap-2 h-16 px-4 py-3 rounded-2xl bg-white/50 backdrop-blur transition-shadow duration-150"
      style={{
        // Gapirayotganda ustunchalar atrofida yengil "porlash" — darajaga
        // qarab kuchayadi/kuchsizlanadi, jim bo'lganda butunlay yo'qoladi.
        boxShadow: isActive
          ? `0 0 ${12 + clamped * 24}px ${2 + clamped * 4}px rgba(147, 51, 234, ${0.18 + clamped * 0.32})`
          : "none",
      }}
      aria-hidden="true"
    >
      {BAR_WEIGHTS.map((weight, i) => {
        const fraction = clamped * weight;
        const height = MIN_HEIGHT_PX + fraction * (MAX_HEIGHT_PX - MIN_HEIGHT_PX);
        const opacity = 0.45 + fraction * 0.55;
        return (
          <div
            key={i}
            className="w-2.5 rounded-full"
            style={{
              height: `${height}px`,
              backgroundColor: lerpColor(MUTED_RGB, VIVID_RGB, fraction),
              opacity,
              transition: `height 150ms ease-out ${BAR_DELAYS_MS[i]}ms, background-color 150ms ease-out ${BAR_DELAYS_MS[i]}ms, opacity 150ms ease-out ${BAR_DELAYS_MS[i]}ms`,
            }}
          />
        );
      })}
    </div>
  );
}

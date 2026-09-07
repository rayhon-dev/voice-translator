export default function AnimatedBackground() {
  return (
    <div className="fixed inset-0 -z-10 overflow-hidden bg-[#FAF8FC]">
      <svg
        className="absolute w-[220%] h-[220%] -top-1/2 -left-1/2 animate-wave-drift"
        viewBox="0 0 1000 1000"
        preserveAspectRatio="none"
      >
        <defs>
          <linearGradient id="wave1" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#E9D5FF" />
            <stop offset="50%" stopColor="#FBCFE8" />
            <stop offset="100%" stopColor="#BFDBFE" />
          </linearGradient>
          <linearGradient id="wave2" x1="100%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#BFDBFE" />
            <stop offset="50%" stopColor="#DDD6FE" />
            <stop offset="100%" stopColor="#FBCFE8" />
          </linearGradient>
          <linearGradient id="wave3" x1="0%" y1="100%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#FBCFE8" />
            <stop offset="100%" stopColor="#E9D5FF" />
          </linearGradient>
          <filter id="softBlur" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="35" />
          </filter>
        </defs>
        <g filter="url(#softBlur)">
          <path
            d="M0,350 C250,250 350,500 600,400 C800,330 900,450 1000,370 L1000,1000 L0,1000 Z"
            fill="url(#wave1)"
            opacity="0.6"
          />
          <path
            d="M0,550 C200,650 400,450 650,550 C820,610 900,500 1000,570 L1000,1000 L0,1000 Z"
            fill="url(#wave2)"
            opacity="0.55"
          />
          <path
            d="M0,750 C300,820 500,700 750,780 C860,810 930,760 1000,790 L1000,1000 L0,1000 Z"
            fill="url(#wave3)"
            opacity="0.45"
          />
        </g>
      </svg>
    </div>
  );
}
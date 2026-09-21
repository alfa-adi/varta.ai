import React, { useEffect, useState } from 'react';
import { cn } from '../../utils/cn';

export function Waveform({ active }: { active: boolean }) {
  const [bars, setBars] = useState<number[]>(Array(24).fill(20));

  useEffect(() => {
    if (!active) {
      setBars(Array(24).fill(20));
      return;
    }

    const interval = setInterval(() => {
      setBars(prev => prev.map(() => Math.floor(Math.random() * 60) + 20));
    }, 150);

    return () => clearInterval(interval);
  }, [active]);

  return (
    <div className="flex items-center justify-center gap-1.5 h-24">
      {bars.map((height, i) => (
        <div 
          key={i}
          className={cn(
            "w-3 rounded-full transition-all duration-150 ease-out",
            active ? "bg-brand-primary" : "bg-text-tertiary/30"
          )}
          style={{ 
            height: `${height}px`,
            opacity: active ? 0.7 + (Math.random() * 0.3) : 0.5
          }}
        />
      ))}
    </div>
  );
}

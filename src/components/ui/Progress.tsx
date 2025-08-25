import { clsx } from "@/lib/utils";
import { JSX } from "react";

export function Progress({ value, className }: { value: number; className?: string }): JSX.Element {
    const clamped = Math.min(100, Math.max(0, value))
    return (
      <div
        className={clsx(
          "w-full h-2 bg-gray-100 rounded-full overflow-hidden",
          className
        )}
        aria-label="progress"
        aria-valuenow={clamped}
      >
        <div
          className="h-full bg-[var(--primary-color)]"
          style={{
            width: `${clamped}%`,
          }}
        />
      </div>
    );
}
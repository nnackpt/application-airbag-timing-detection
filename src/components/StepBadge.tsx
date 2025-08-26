import { FC } from "react"

interface StepBadgeProps {
    label: string
    active: boolean
    done: boolean
}

export const StepBadge: FC<StepBadgeProps> = ({ label, active, done }) => {
    return (
        <div className="flex items-center gap-3">
            <div
                className={[
                    "flex h-7 w-7 items-center justify-center rounded-full border text-xs font-bold",
                    done ? "border-[#005496] bg-[#005496] text-white" : active ? "border-[#005496] text-[#005496]" : "border-gray-300 text-gray-400"
                ].join(" ")}
            >
                {done ? "✓" : active ? "•" : ""}
            </div>
            <span className={done ? "text-[#005496]" : active ? "text-gray-800" : "text-gray-400"}>{label}</span>
        </div>
    )
}
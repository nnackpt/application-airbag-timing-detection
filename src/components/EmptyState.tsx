import Link from "next/link"
import { JSX } from "react"

export function EmptyState({ 
    title, 
    actionHref, 
    actionLabel 
}: { 
    title: string
    actionHref: string
    actionLabel: string
}): JSX.Element {
    return (
        <div className="text-center p-10 border-2 border-dashed rounded-2xl">
            <p className="text-gray-600 mb-3">{title}</p>
            {actionHref && actionLabel ? (
                <Link 
                    href={actionHref}
                    className="inline-flex items-center justify-center gap-2 rounded-2xl px-4 py-2 font-medium transition 
                    border border-[var(--primary-color)] text-[var(--primary-color)] hover:bg-[var(--primary-color)]/10"
                >
                    {actionLabel}
                </Link>
            ): null}
        </div>
    )
}
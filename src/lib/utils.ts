export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"

export function fmtDate(input: string | null): string {
    if (!input) return "-"
    const d = new Date(input)
    return d.toLocaleString(undefined, { hour12: false })
}

export function getSearchParam(key: string): string | null {
    if (typeof window === "undefined") return null
    const url = new URL(window.location.href)
    return url.searchParams.get(key)
}

export function clsx(...classes: Array<string | false | undefined>): string {
    return classes.filter(Boolean).join(" ")
}
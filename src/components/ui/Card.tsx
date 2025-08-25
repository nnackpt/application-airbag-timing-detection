import { JSX, PropsWithChildren } from "react";

export function Card({ children }: PropsWithChildren): JSX.Element {
    return <div className="bg-white rounded-2xl shadow-soft border border-gray-100 p-5">{children}</div>
}

export function CardHeader({ children }: PropsWithChildren): JSX.Element {
    return <div className="mb-4 flex items-center justify-between gap-2">{children}</div>
}

export function CardTitle({ children }: PropsWithChildren): JSX.Element {
    return <h3 className="h3">{children}</h3>
}

export function CardContent({ children }: PropsWithChildren): JSX.Element {
    return <div className="space-y-3">{children}</div>
}
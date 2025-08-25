"use client"

import { clsx } from "@/lib/utils";
import { ButtonHTMLAttributes, JSX, PropsWithChildren } from "react";

export default function Button({ className, children, ...props }: PropsWithChildren<ButtonHTMLAttributes<HTMLButtonElement>>): JSX.Element {
    return (
        <button 
            className={clsx
                ("inline-flex items-center justify-center gap-2 rounded-2xl px-4 py-2 font-medium transition bg-[var(--primary-color)] text-white hover:opacity-90 active:opacity-80", className)
            } {...props}
        >
            {children}
        </button>
    )
}
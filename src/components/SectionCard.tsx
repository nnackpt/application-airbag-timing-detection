import { FC, ReactNode } from "react";

export const SectionCard: FC<{ title: string; children: ReactNode; right?: ReactNode }> = ({ title, right, children }) => {
    return (
        <section className="mb-6 rounded-2xl border border-gray-10 bg-white p-5 shadow-soft">
            <div className="mb-4 flex items-center justify-between">
                <h2 className="text-lg font-semibold text-gray-800">{title}</h2>
                {right}
            </div>
            {children}
        </section>
    )
}
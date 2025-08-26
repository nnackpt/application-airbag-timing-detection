import { FC } from "react";

export const VideoPlayer: FC<{ src: string; filename?: string }> = ({ src, filename }) => (
    <div className="rounded-xl border border-gray-100 p-3">
        <video src={src} controls className="h-auto w-full rounded-lg" />
        {filename && (
            <div className="mt-2 text-xs text-gray-500">{filename}</div>
        )}
        <div className="mt-3">
            <a 
                href={src}
                className="inline-flex items-center rounded-lg bg-[#005496] px-4 py-2 text-sm font-medium text-white hover:opacity-90"
                download
            >
                Download Video
            </a>
        </div>
    </div>
)
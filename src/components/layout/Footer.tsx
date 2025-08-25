import { JSX } from "react";

export default function Footer(): JSX.Element {
    return (
      <footer className="mt-16 border-t border-gray-100">
        <div className="mx-auto max-w-6xl px-4 py-8 text-xs text-gray-500 flex items-center justify-between">
          <p>© {new Date().getFullYear()} Airbag Timing Detection. All rights reserved.</p>
        </div>
      </footer>
    );
}
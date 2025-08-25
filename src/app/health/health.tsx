import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { getHealth } from "@/lib/api";
import { JSX } from "react";

export default async function Health(): Promise<JSX.Element> {
    const data = await getHealth()

    return (
        <Card>
            <CardHeader>
                <CardTitle>
                    System Health
                </CardTitle>
            </CardHeader>
            <CardContent>
                <div className="grid md:grid-cols-2 gap-6">
                    <div>
                        <ul className="text-sm text-gray-700 space-y-1">
                            <li><span className="font-medium">Status:</span> {data.status}</li>
                            <li><span className="font-medium">Version:</span> {data.version}</li>
                            <li><span className="font-medium">Timestamp:</span> {new Date(data.timestamp).toLocaleString()}</li>
                            <li><span className="font-medium">Storage:</span> {data.storage.type}</li>
                            <li><span className="font-medium">Total videos:</span> {data.storage.total_videos}</li>
                            <li><span className="font-medium">Active tasks:</span> {data.storage.active_tasks}</li>
                        </ul>
                    </div>
                    <div>
                        <h4 className="text-xl md:text-2xl font-semibold mb-2">Models</h4>
                        <div className="flex flex-wrap gap-2">
                            {Object.entries(data.models_loaded).map(([k, v]) => (
                                <span 
                                    key={k} 
                                    className={v 
                                        ? "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium border bg-green-50 text-green-700 border-green-200" 
                                        : "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium border bg-amber-50 text-amber-700 border-amber-200"
                                    }
                                >
                                    {k}: {v ? "loaded" : "missing"}
                                </span>
                            ))}
                            <h4 className="text-xl md:text-2xl font-semibold mt-6 mb-2">Enhancements</h4>
                            <ul className="list-disc list-inside text-sm text-gray-700 space-y-1">
                                {data.enhancements.map((e) => (<li key={e}>{e}</li>))}
                            </ul>
                        </div>
                    </div>
                </div>
            </CardContent>
        </Card>
    )
}
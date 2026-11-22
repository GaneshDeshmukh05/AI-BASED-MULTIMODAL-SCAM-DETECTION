
import { useState } from "react";
import {
    scanMultimodal,
} from "./api";
import {
    ShieldCheck,
    ShieldAlert,
    Upload,
    Link,
    MessageSquareText,
    AudioLines,
    Image,
    LoaderCircle,
    AlertTriangle,
    CheckCircle2,
    ArrowLeft,
} from "lucide-react";

export default function ScanCentre({ onBack }) {
    const [text, setText] = useState("");
    const [url, setUrl] = useState("");
    const [audioFile, setAudioFile] = useState(null);
    const [screenshotFile, setScreenshotFile] = useState(null);
    const [offlineMode, setOfflineMode] = useState(() => {
        return localStorage.getItem("scamshield_offline_mode") !== "false";
    });
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState("");

    const handleScan = async (event) => {
        event.preventDefault();
        setError("");
        setResult(null);

        if (
            !text.trim() &&
            !url.trim() &&
            !audioFile &&
            !screenshotFile
        ) {
            setError("Please provide at least one input to scan.");
            return;
        }

        if (audioFile && audioFile.size > 15 * 1024 * 1024) {
            setError("Audio file must be smaller than 15 MB.");
            return;
        }

        if (screenshotFile && screenshotFile.size > 15 * 1024 * 1024) {
            setError("Screenshot must be smaller than 15 MB.");
            return;
        }

        setLoading(true);

        try {
            const data = await scanMultimodal({
                text,
                url,
                audioFile,
                screenshotFile,
                offlineMode,
            });

            setResult(data);
        } catch (err) {
            setError(
                err.response?.data?.detail ||
                err.message ||
                "Scan failed. Please check the backend."
            );
        } finally {
            setLoading(false);
        }
    };

    const risk = Math.round((result?.overall_risk_score ?? 0) * 100);
    const confidence = Math.round((result?.overall_confidence ?? 0) * 100);
    const verdict = result?.verdict?.toUpperCase() || "UNKNOWN";
    const isThreat = ["SCAM", "MALICIOUS", "PHISHING"].includes(verdict);

    const panel =
        "rounded-xl border border-slate-800 bg-[#111827] p-5";
    const field =
        "mt-2 w-full rounded-lg border border-slate-700 bg-[#0b1220] px-4 py-3 text-sm text-white outline-none focus:border-blue-500";

    return (
        <div className="min-h-screen bg-[#080d19] p-6 text-slate-200">
            <div className="mx-auto max-w-6xl">
                <button
                    onClick={onBack}
                    className="mb-6 flex items-center gap-2 text-sm text-slate-400 hover:text-white"
                >
                    <ArrowLeft size={17} /> Back to Dashboard
                </button>

                <div className="mb-8">
                    <div className="flex items-center gap-3">
                        <div className="rounded-xl bg-blue-500/15 p-3 text-blue-400">
                            <ShieldCheck size={26} />
                        </div>
                        <div>
                            <h1 className="text-2xl font-bold text-white">
                                Scan Centre
                            </h1>
                            <p className="text-sm text-slate-400">
                                Analyze suspicious messages, links, screenshots and audio.
                            </p>
                        </div>
                    </div>
                </div>

                <form onSubmit={handleScan} className="space-y-5">
                    <div className="grid gap-5 md:grid-cols-2">
                        <section className={panel}>
                            <div className="flex items-center gap-2">
                                <MessageSquareText className="text-blue-400" size={20} />
                                <h2 className="font-semibold text-white">Text Analysis</h2>
                            </div>
                            <p className="mt-2 text-xs text-slate-400">
                                Paste a suspicious SMS, email or message.
                            </p>
                            <textarea
                                value={text}
                                onChange={(e) => setText(e.target.value)}
                                placeholder="Paste suspicious message here..."
                                rows={6}
                                className={field + " resize-y"}
                            />
                        </section>

                        <section className={panel}>
                            <div className="flex items-center gap-2">
                                <Link className="text-purple-400" size={20} />
                                <h2 className="font-semibold text-white">URL Analysis</h2>
                            </div>
                            <p className="mt-2 text-xs text-slate-400">
                                Enter a suspicious website address.
                            </p>
                            <input
                                type="text"
                                value={url}
                                onChange={(e) => setUrl(e.target.value)}
                                placeholder="https://example.com"
                                className={field}
                            />
                            <p className="mt-3 text-xs text-slate-500">
                                ScamShield analyzes the URL. It will not open the website.
                            </p>
                        </section>

                        <section className={panel}>
                            <div className="flex items-center gap-2">
                                <Image className="text-cyan-400" size={20} />
                                <h2 className="font-semibold text-white">Screenshot Analysis</h2>
                            </div>
                            <p className="mt-2 text-xs text-slate-400">
                                Upload a screenshot of a suspicious conversation or page.
                            </p>
                            <label className="mt-4 flex cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed border-slate-600 bg-[#0b1220] p-6 text-center hover:border-blue-500">
                                <Upload className="mb-2 text-slate-400" size={25} />
                                <span className="text-sm text-slate-300">
                                    {screenshotFile?.name || "Choose PNG or JPG"}
                                </span>
                                <span className="mt-1 text-xs text-slate-500">
                                    Maximum size: 15 MB
                                </span>
                                <input
                                    type="file"
                                    accept=".png,.jpg,.jpeg,image/png,image/jpeg"
                                    className="hidden"
                                    onChange={(e) =>
                                        setScreenshotFile(e.target.files?.[0] || null)
                                    }
                                />
                            </label>
                            {screenshotFile && (
                                <button
                                    type="button"
                                    onClick={() => setScreenshotFile(null)}
                                    className="mt-2 text-xs text-red-400"
                                >
                                    Remove file
                                </button>
                            )}
                        </section>

                        <section className={panel}>
                            <div className="flex items-center gap-2">
                                <AudioLines className="text-pink-400" size={20} />
                                <h2 className="font-semibold text-white">Audio Analysis</h2>
                            </div>
                            <p className="mt-2 text-xs text-slate-400">
                                Upload a WAV recording for scam-call analysis.
                            </p>
                            <label className="mt-4 flex cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed border-slate-600 bg-[#0b1220] p-6 text-center hover:border-blue-500">
                                <Upload className="mb-2 text-slate-400" size={25} />
                                <span className="text-sm text-slate-300">
                                    {audioFile?.name || "Choose WAV audio"}
                                </span>
                                <span className="mt-1 text-xs text-slate-500">
                                    Maximum size: 15 MB
                                </span>
                                <input
                                    type="file"
                                    accept=".wav,audio/wav,audio/x-wav"
                                    className="hidden"
                                    onChange={(e) =>
                                        setAudioFile(e.target.files?.[0] || null)
                                    }
                                />
                            </label>
                            {audioFile && (
                                <button
                                    type="button"
                                    onClick={() => setAudioFile(null)}
                                    className="mt-2 text-xs text-red-400"
                                >
                                    Remove file
                                </button>
                            )}
                        </section>
                    </div>

                    <section className={panel}>
                        <label className="flex cursor-pointer items-start gap-3">
                            <input
                                type="checkbox"
                                checked={offlineMode}
                                onChange={(e) => setOfflineMode(e.target.checked)}
                                className="mt-1 accent-blue-500"
                            />
                            <span>
                                <span className="block text-sm font-medium text-white">
                                    Offline Mode
                                </span>
                                <span className="text-xs text-slate-400">
                                    Enabled by default. Disabling it allows domain intelligence
                                    checks where supported.
                                </span>
                            </span>
                        </label>
                    </section>

                    {error && (
                        <div className="flex items-start gap-2 rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">
                            <AlertTriangle size={18} />
                            {error}
                        </div>
                    )}

                    <button
                        type="submit"
                        disabled={loading}
                        className="flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 px-6 py-4 font-semibold text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                        {loading ? (
                            <>
                                <LoaderCircle className="animate-spin" size={20} />
                                Analyzing...
                            </>
                        ) : (
                            <>
                                <ShieldCheck size={20} />
                                Analyze for Scams
                            </>
                        )}
                    </button>
                </form>

                {result && (
                    <section className="mt-8 space-y-5">
                        <div className={panel}>
                            <div className="flex flex-wrap items-center justify-between gap-4">
                                <div>
                                    <p className="text-xs uppercase tracking-wider text-slate-400">
                                        Analysis Result
                                    </p>
                                    <div className="mt-2 flex items-center gap-3">
                                        {isThreat ? (
                                            <ShieldAlert className="text-red-400" size={30} />
                                        ) : (
                                            <CheckCircle2 className="text-emerald-400" size={30} />
                                        )}
                                        <h2 className="text-2xl font-bold text-white">
                                            {verdict}
                                        </h2>
                                    </div>
                                    <p className="mt-2 text-sm text-slate-400">
                                        Risk level: {result.risk_level || "Unknown"}
                                    </p>
                                </div>

                                <div className="flex gap-8">
                                    <div>
                                        <p className="text-xs text-slate-400">Risk Score</p>
                                        <p className="text-3xl font-bold text-white">{risk}%</p>
                                    </div>
                                    <div>
                                        <p className="text-xs text-slate-400">Confidence</p>
                                        <p className="text-3xl font-bold text-white">
                                            {confidence}%
                                        </p>
                                    </div>
                                </div>
                            </div>

                            <div className="mt-5 h-2 overflow-hidden rounded-full bg-slate-800">
                                <div
                                    className="h-full rounded-full bg-gradient-to-r from-emerald-400 via-yellow-400 to-red-500"
                                    style={{ width: `${Math.min(100, Math.max(0, risk))}%` }}
                                />
                            </div>

                            {result.explanation?.headline && (
                                <h3 className="mt-5 font-semibold text-white">
                                    {result.explanation.headline}
                                </h3>
                            )}
                            {result.explanation?.why && (
                                <p className="mt-2 text-sm leading-6 text-slate-300">
                                    {result.explanation.why}
                                </p>
                            )}
                        </div>

                        {result.has_conflict && (
                            <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-5">
                                <h3 className="font-semibold text-amber-300">
                                    Conflicting Model Results
                                </h3>
                                <p className="mt-2 text-sm text-slate-300">
                                    {result.conflict_details || "The analysis detected conflicting signals between modalities."}
                                </p>
                            </div>
                        )}

                        <div className={panel}>
                            <h3 className="mb-4 font-semibold text-white">Evidence Found</h3>
                            {result.explanation?.evidence_citations?.length ? (
                                <ul className="space-y-3">
                                    {result.explanation.evidence_citations.map((item, index) => (
                                        <li key={index} className="flex gap-2 text-sm text-slate-300">
                                            <AlertTriangle
                                                size={16}
                                                className="mt-0.5 shrink-0 text-amber-400"
                                            />
                                            {item}
                                        </li>
                                    ))}
                                </ul>
                            ) : (
                                <p className="text-sm text-slate-400">
                                    No evidence citations were returned.
                                </p>
                            )}
                        </div>

                        <div className={panel}>
                            <h3 className="mb-4 font-semibold text-white">Recommended Actions</h3>
                            {result.explanation?.recommended_actions?.length ? (
                                <ul className="space-y-3">
                                    {result.explanation.recommended_actions.map((item, index) => (
                                        <li key={index} className="flex gap-2 text-sm text-slate-300">
                                            <CheckCircle2
                                                size={16}
                                                className="mt-0.5 shrink-0 text-blue-400"
                                            />
                                            {item}
                                        </li>
                                    ))}
                                </ul>
                            ) : (
                                <p className="text-sm text-slate-400">
                                    No recommended actions were returned.
                                </p>
                            )}
                        </div>

                        <div className={panel}>
                            <h3 className="mb-4 font-semibold text-white">Modality Results</h3>
                            <div className="grid gap-3 sm:grid-cols-2">
                                {Object.entries(result.modalities || {}).map(([name, data]) => (
                                    <div
                                        key={name}
                                        className="rounded-lg border border-slate-700 bg-[#0b1220] p-4"
                                    >
                                        <div className="flex items-center justify-between">
                                            <span className="font-medium capitalize text-white">
                                                {name}
                                            </span>
                                            <span className="text-xs text-slate-400">
                                                {data.status || "Unknown"}
                                            </span>
                                        </div>
                                        <p className="mt-2 text-sm text-slate-300">
                                            Result: {data.label || "N/A"}
                                        </p>
                                        <p className="mt-1 text-sm text-slate-400">
                                            Risk: {Math.round((data.risk_score ?? 0) * 100)}%
                                        </p>
                                        {data.error_message && (
                                            <p className="mt-2 text-xs text-red-400">
                                                {data.error_message}
                                            </p>
                                        )}
                                    </div>
                                ))}
                            </div>
                            {!!result.missing_modalities?.length && (
                                <p className="mt-4 text-xs text-slate-400">
                                    Not provided: {result.missing_modalities.join(", ")}
                                </p>
                            )}
                            {!!result.failed_modalities?.length && (
                                <p className="mt-2 text-xs text-red-400">
                                    Failed: {result.failed_modalities.join(", ")}
                                </p>
                            )}
                        </div>

                        <button
                            type="button"
                            onClick={() => {
                                setResult(null);
                                setError("");
                            }}
                            className="rounded-lg border border-slate-700 px-5 py-3 text-sm text-slate-300 hover:bg-slate-800"
                        >
                            New Scan
                        </button>
                    </section>
                )}
            </div>
        </div>
    );
}

import { useEffect, useState } from "react";
import { getScanDetails, getScanHistory } from "./api";
import { History, RefreshCw, Search, ShieldAlert, ShieldCheck, LoaderCircle, AlertTriangle, ChevronDown, ChevronUp, ArrowLeft } from "lucide-react";

const dateLabel = (value) => {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
};

const verdictStyle = (value) => {
  const v = (value || "").toUpperCase();
  if (["SCAM", "MALICIOUS", "PHISHING"].includes(v)) return "bg-red-500/10 text-red-400";
  if (["SAFE", "LEGITIMATE"].includes(v)) return "bg-emerald-500/10 text-emerald-400";
  return "bg-amber-500/10 text-amber-400";
};

export default function ScanHistory({ onBack, onNewScan }) {
  const [scans, setScans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState(null);
  const [details, setDetails] = useState(null);
  const [detailsLoading, setDetailsLoading] = useState(false);

  const loadHistory = async () => {
    setLoading(true);
    setError("");
    try {
      const data = await getScanHistory(100);
      setScans(data.scans || []);
    } catch (err) {
      setError(err.response?.data?.detail || "Could not load history. Check the backend is running and updated.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadHistory(); }, []);

  const toggleDetails = async (id) => {
    if (selectedId === id) {
      setSelectedId(null);
      setDetails(null);
      return;
    }
    setSelectedId(id);
    setDetails(null);
    setDetailsLoading(true);
    try {
      setDetails(await getScanDetails(id));
    } catch (err) {
      setError(err.response?.data?.detail || "Could not load scan details.");
    } finally {
      setDetailsLoading(false);
    }
  };

  const filtered = scans.filter((scan) => [
    scan.verdict, scan.risk_level, scan.inputs?.url, scan.inputs?.text,
    scan.inputs?.audio_filename, scan.inputs?.screenshot_filename
  ].filter(Boolean).join(" ").toLowerCase().includes(query.toLowerCase()));

  return (
    <div className="min-h-screen bg-[#080b14] p-5 text-slate-100 sm:p-8">
      <div className="mx-auto max-w-6xl">
        <div className="mb-7 flex flex-wrap items-end justify-between gap-4">
          <div>
            <button onClick={onBack} className="mb-4 flex items-center gap-2 text-sm text-slate-400 hover:text-white">
              <ArrowLeft size={16} /> Back to Dashboard
            </button>
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-blue-500/10 p-3 text-blue-400"><History size={24} /></div>
              <div>
                <h1 className="text-2xl font-bold">Scan History</h1>
                <p className="mt-1 text-sm text-slate-400">Saved analysis results from your SQLite database.</p>
              </div>
            </div>
          </div>
          <div className="flex gap-2">
            <button onClick={loadHistory} disabled={loading} className="flex items-center gap-2 rounded-lg border border-slate-700 px-4 py-2.5 text-sm text-slate-300 hover:bg-white/5 disabled:opacity-50">
              <RefreshCw size={16} className={loading ? "animate-spin" : ""} /> Refresh
            </button>
            <button onClick={onNewScan} className="rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-500">New Scan</button>
          </div>
        </div>

        <div className="mb-5 flex flex-col gap-3 rounded-xl border border-white/10 bg-[#101626] p-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="relative w-full sm:max-w-md">
            <Search size={17} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search verdict, URL, text or filename..." className="w-full rounded-lg border border-slate-700 bg-[#0b1220] py-2.5 pl-10 pr-3 text-sm text-white outline-none focus:border-blue-500" />
          </div>
          <p className="text-sm text-slate-400">{filtered.length} of {scans.length} saved scans</p>
        </div>

        {error && <div className="mb-5 flex gap-2 rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300"><AlertTriangle size={18} />{error}</div>}

        {loading ? (
          <div className="flex items-center justify-center gap-3 rounded-xl border border-white/10 bg-[#101626] p-12 text-slate-400"><LoaderCircle size={20} className="animate-spin" /> Loading saved scans...</div>
        ) : filtered.length === 0 ? (
          <div className="rounded-xl border border-white/10 bg-[#101626] p-12 text-center">
            <History size={34} className="mx-auto mb-3 text-slate-500" />
            <h2 className="font-semibold text-white">{scans.length === 0 ? "No saved scans yet" : "No matching scans"}</h2>
            <p className="mt-2 text-sm text-slate-400">{scans.length === 0 ? "Run a new analysis. Its result will automatically appear here." : "Try another search term."}</p>
          </div>
        ) : (
          <div className="space-y-3">
            {filtered.map((scan) => {
              const inputLabel = scan.inputs?.url || scan.inputs?.text || scan.inputs?.screenshot_filename || scan.inputs?.audio_filename || "Multimodal scan";
              const expanded = selectedId === scan.scan_id;
              const threat = ["SCAM", "MALICIOUS", "PHISHING"].includes((scan.verdict || "").toUpperCase());
              return (
                <div key={scan.scan_id} className="overflow-hidden rounded-xl border border-white/10 bg-[#101626]">
                  <button onClick={() => toggleDetails(scan.scan_id)} className="flex w-full flex-col gap-4 p-4 text-left hover:bg-white/[0.03] sm:flex-row sm:items-center sm:justify-between">
                    <div className="flex min-w-0 items-center gap-3">
                      <div className="rounded-lg bg-white/5 p-2.5 text-slate-400">{threat ? <ShieldAlert size={19} /> : <ShieldCheck size={19} />}</div>
                      <div className="min-w-0">
                        <p className="truncate text-sm font-medium text-white">{inputLabel}</p>
                        <p className="mt-1 text-xs text-slate-500">{dateLabel(scan.created_at)}</p>
                      </div>
                    </div>
                    <div className="flex flex-wrap items-center gap-3">
                      <span className={`rounded-lg px-3 py-1.5 text-xs font-semibold ${verdictStyle(scan.verdict)}`}>{scan.verdict || "UNKNOWN"}</span>
                      <span className="text-sm text-slate-300">Risk: {Math.round((scan.overall_risk_score || 0) * 100)}%</span>
                      {expanded ? <ChevronUp size={17} /> : <ChevronDown size={17} />}
                    </div>
                  </button>
                  {expanded && <div className="space-y-4 border-t border-white/10 p-5">
                    {detailsLoading ? <p className="flex items-center gap-2 text-sm text-slate-400"><LoaderCircle size={16} className="animate-spin" /> Loading details...</p> :
                      details ? <>
                        <div><h3 className="mb-2 text-sm font-semibold text-white">Analysis explanation</h3><p className="text-sm leading-6 text-slate-300">{details.result?.explanation?.why || "No explanation returned."}</p></div>
                        <div><h3 className="mb-2 text-sm font-semibold text-white">Evidence</h3><ul className="space-y-2">{(details.result?.explanation?.evidence_citations || []).map((item, i) => <li key={i} className="text-sm text-slate-300">• {item}</li>)}</ul></div>
                        <div><h3 className="mb-2 text-sm font-semibold text-white">Recommended actions</h3><ul className="space-y-2">{(details.result?.explanation?.recommended_actions || []).map((item, i) => <li key={i} className="text-sm text-slate-300">• {item}</li>)}</ul></div>
                        <p className="text-xs text-slate-500">Scan ID: {details.scan_id}</p>
                      </> : <p className="text-sm text-slate-400">Details could not be loaded.</p>}
                  </div>}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

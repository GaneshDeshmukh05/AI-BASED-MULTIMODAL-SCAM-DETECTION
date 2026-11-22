
import { useEffect, useMemo, useState } from "react";
import { checkBackendHealth } from "./api";
import ScanCentre from "./ScanCentre";
import ScanHistory from "./ScanHistory";

import {
  ShieldCheck, LayoutDashboard, ScanSearch, History,
  ShieldAlert, Shield, Activity, CircleCheck,
  Clock, RefreshCw, AlertTriangle,
} from "lucide-react";

import {
  ResponsiveContainer, Tooltip, PieChart, Pie, Cell,
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
} from "recharts";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const COLORS = {
  SCAM: "#ef4444",
  SAFE: "#22c55e",
  SUSPICIOUS: "#f59e0b",
  UNKNOWN: "#64748b",
};

function formatDate(value) {
  if (!value) return "Unknown time";

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;

  return date.toLocaleString();
}

function getScanName(scan) {
  const inputs = scan.inputs || {};

  if (inputs.text) return inputs.text;
  if (inputs.url) return inputs.url;
  if (inputs.screenshot_filename) return inputs.screenshot_filename;
  if (inputs.audio_filename) return inputs.audio_filename;

  return "Multimodal scan";
}

function getScanType(scan) {
  const inputs = scan.inputs || {};
  const types = [];

  if (inputs.text) types.push("Text");
  if (inputs.url) types.push("URL");
  if (inputs.screenshot_filename) types.push("Screenshot");
  if (inputs.audio_filename) types.push("Audio");

  return types.length ? types.join(" + ") : "Unknown";
}

function App() {
  const [backendStatus, setBackendStatus] = useState("checking");
  const [activePage, setActivePage] = useState("dashboard");

  const [dashboard, setDashboard] = useState(null);
  const [loadingStats, setLoadingStats] = useState(true);
  const [statsError, setStatsError] = useState("");

  const loadDashboard = async () => {
    try {
      setStatsError("");

      const response = await fetch(`${API_URL}/api/dashboard/stats`);

      if (!response.ok) {
        throw new Error("Could not retrieve dashboard statistics.");
      }

      const data = await response.json();
      setDashboard(data);
      setBackendStatus("connected");
    } catch (error) {
      console.error("Dashboard loading failed:", error);
      setStatsError(error.message);
      setBackendStatus("disconnected");
    } finally {
      setLoadingStats(false);
    }
  };

  useEffect(() => {
    let active = true;

    checkBackendHealth()
      .then(() => {
        if (active) setBackendStatus("connected");
      })
      .catch(() => {
        if (active) setBackendStatus("disconnected");
      });

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    loadDashboard();

    const interval = setInterval(loadDashboard, 30000);
    return () => clearInterval(interval);
  }, []);

  // Refresh statistics whenever the user returns to the dashboard.
  useEffect(() => {
    if (activePage === "dashboard") {
      loadDashboard();
    }
  }, [activePage]);

  const totalScans = dashboard?.total_scans ?? 0;
  const scamCount = dashboard?.scam_detections ?? 0;
  const safeCount = dashboard?.safe_detections ?? 0;
  const averageRisk = dashboard?.average_risk ?? 0;

  const detectionRate = totalScans
    ? (scamCount / totalScans) * 100
    : 0;

  const riskScore = Math.round(averageRisk * 100);

  const distribution = useMemo(() => {
    const rows = dashboard?.verdict_distribution || [];

    return rows.map((row) => {
      const verdict = (row.verdict || "UNKNOWN").toUpperCase();

      return {
        name: verdict,
        value: row.count,
        color: COLORS[verdict] || COLORS.UNKNOWN,
      };
    });
  }, [dashboard]);

  const dailyActivity = useMemo(() => {
    return (dashboard?.daily_activity || []).map((day) => ({
      ...day,
      label: new Date(`${day.date}T00:00:00`).toLocaleDateString(
        undefined,
        { weekday: "short" }
      ),
    }));
  }, [dashboard]);

  const recentScans = dashboard?.recent_scans || [];

  const statCards = [
    {
      title: "Total Scans",
      value: totalScans.toLocaleString(),
      icon: ScanSearch,
      color: "blue",
      description: "All saved analyses",
    },
    {
      title: "Threats Detected",
      value: scamCount.toLocaleString(),
      icon: ShieldAlert,
      color: "red",
      description: "Scam verdicts",
    },
    {
      title: "Safe Results",
      value: safeCount.toLocaleString(),
      icon: ShieldCheck,
      color: "green",
      description: "Safe verdicts",
    },
    {
      title: "Detection Rate",
      value: `${detectionRate.toFixed(1)}%`,
      icon: Activity,
      color: "purple",
      description: "Scam detections / total scans",
    },
  ];

  const colorClasses = {
    blue: "bg-blue-500/10 text-blue-400",
    red: "bg-red-500/10 text-red-400",
    green: "bg-green-500/10 text-green-400",
    purple: "bg-purple-500/10 text-purple-400",
  };

  return (
    <div className="min-h-screen bg-[#080b14] text-slate-100">

      {/* Sidebar */}
      <aside className="fixed inset-y-0 left-0 z-20 hidden w-64 border-r border-white/10 bg-[#0b1020] lg:flex lg:flex-col">
        <div className="flex h-20 items-center gap-3 border-b border-white/10 px-6">
          <div className="rounded-xl bg-blue-500/15 p-2.5 text-blue-400">
            <ShieldCheck size={25} />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-wide">
              ScamShield <span className="text-blue-400">AI</span>
            </h1>
            <p className="text-xs text-slate-500">Multimodal Protection</p>
          </div>
        </div>

        <nav className="flex-1 space-y-2 px-4 py-7">
          <p className="mb-4 px-3 text-xs font-semibold uppercase tracking-widest text-slate-500">
            Workspace
          </p>

          {[
            { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
            { id: "scan", label: "Scan Centre", icon: ScanSearch },
            { id: "history", label: "Scan History", icon: History },
          ].map((item) => {
            const Icon = item.icon;

            return (
              <button
                key={item.id}
                onClick={() => setActivePage(item.id)}
                className={`flex w-full items-center gap-3 rounded-xl px-4 py-3 text-sm font-medium transition ${activePage === item.id
                  ? "border border-blue-400/20 bg-blue-500/10 text-blue-400"
                  : "text-slate-400 hover:bg-white/5 hover:text-white"
                  }`}
              >
                <Icon size={19} />
                {item.label}
              </button>
            );
          })}

        </nav>

        <div className="m-4 rounded-xl border border-blue-400/15 bg-blue-500/5 p-4">
          <div className="mb-2 flex items-center gap-2 text-sm font-medium text-blue-300">
            <Shield size={17} /> Protection Active
          </div>
          <p className="text-xs leading-5 text-slate-500">
            Your security monitoring dashboard is ready.
          </p>
          <div className="mt-3 flex items-center gap-2 text-xs text-green-400">
            <span className="h-2 w-2 rounded-full bg-green-400" />
            Live dashboard
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="lg:pl-64">
        <header className="flex h-20 items-center justify-between border-b border-white/10 px-5 sm:px-8">
          <div>
            <p className="text-sm text-slate-500">Workspace / Overview</p>
            <h2 className="text-lg font-semibold">
              {activePage === "dashboard"
                ? "Dashboard"
                : activePage === "scan"
                  ? "Scan Centre"
                  : "Scan History"}
            </h2>
          </div>

          <div className="flex items-center gap-4">
            <button
              onClick={loadDashboard}
              title="Refresh dashboard"
              className="rounded-xl border border-white/10 p-2.5 text-slate-400 hover:bg-white/5"
            >
              <RefreshCw size={18} />
            </button>

          </div>
        </header>

        {activePage === "scan" ? (
          <ScanCentre onBack={() => setActivePage("dashboard")} />
        ) : activePage === "history" ? (
          <ScanHistory
            onBack={() => setActivePage("dashboard")}
            onNewScan={() => setActivePage("scan")}
          />
        ) : (
          <div className="space-y-8 p-5 sm:p-8">

            {/* Overview heading */}
            <section className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
              <div>
                <h2 className="text-2xl font-bold">Security Overview</h2>
                <p className="mt-1 text-sm text-slate-400">
                  Live statistics from your saved scam analyses.
                </p>
              </div>

              <button
                onClick={() => setActivePage("scan")}
                className="flex items-center justify-center gap-2 rounded-xl bg-blue-600 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-blue-600/20 transition hover:bg-blue-500"
              >
                <ScanSearch size={18} /> Start New Scan
              </button>
            </section>

            {/* Error message */}
            {statsError && (
              <div className="flex items-center gap-3 rounded-xl border border-red-500/20 bg-red-500/5 p-4 text-sm text-red-400">
                <AlertTriangle size={18} />
                {statsError}
                <button
                  onClick={loadDashboard}
                  className="ml-auto underline"
                >
                  Retry
                </button>
              </div>
            )}

            {/* Statistic cards */}
            <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              {statCards.map((stat) => {
                const Icon = stat.icon;

                return (
                  <div
                    key={stat.title}
                    className="rounded-2xl border border-white/10 bg-[#101626] p-5"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-slate-400">
                        {stat.title}
                      </span>
                      <div className={`rounded-xl p-2.5 ${colorClasses[stat.color]}`}>
                        <Icon size={20} />
                      </div>
                    </div>

                    <div className="mt-5">
                      <span className="text-3xl font-bold">
                        {loadingStats ? "..." : stat.value}
                      </span>
                    </div>

                    <p className="mt-2 text-xs text-slate-500">
                      {stat.description}
                    </p>
                  </div>
                );
              })}
            </section>

            {/* Distribution and risk gauge */}
            <section className="grid grid-cols-1 gap-6 xl:grid-cols-2">

              <div className="rounded-2xl border border-white/10 bg-[#101626] p-6">
                <h3 className="font-semibold">Scam Distribution</h3>
                <p className="mt-1 text-xs text-slate-500">
                  Actual saved scan verdicts
                </p>

                {distribution.length ? (
                  <>
                    <div className="mt-4">
                      <ResponsiveContainer width="100%" height={220}>
                        <PieChart>
                          <Pie
                            data={distribution}
                            dataKey="value"
                            nameKey="name"
                            cx="50%"
                            cy="50%"
                            innerRadius={62}
                            outerRadius={88}
                            paddingAngle={5}
                            stroke="none"
                          >
                            {distribution.map((entry) => (
                              <Cell
                                key={entry.name}
                                fill={entry.color}
                              />
                            ))}
                          </Pie>
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "#151e30",
                              border: "1px solid #334155",
                              borderRadius: "12px",
                              color: "#fff",
                            }}
                          />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>

                    <div className="space-y-3">
                      {distribution.map((item) => (
                        <div
                          key={item.name}
                          className="flex items-center justify-between"
                        >
                          <div className="flex items-center gap-2">
                            <span
                              className="h-2.5 w-2.5 rounded-full"
                              style={{ backgroundColor: item.color }}
                            />
                            <span className="text-sm text-slate-400">
                              {item.name}
                            </span>
                          </div>
                          <span className="text-sm font-semibold">
                            {item.value} scans
                          </span>
                        </div>
                      ))}
                    </div>
                  </>
                ) : (
                  <div className="flex h-64 items-center justify-center text-sm text-slate-500">
                    No scan data available yet.
                  </div>
                )}
              </div>

              {/* Average risk gauge */}
              <div className="rounded-2xl border border-white/10 bg-[#101626] p-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-base font-semibold">Average Risk Score</h3>
                  <span className="text-xs text-slate-500">All scans</span>
                </div>

                <div className="mx-auto mt-5 w-full max-w-[420px]">
                  <svg viewBox="0 0 360 260" className="w-full">
                    <defs>
                      <linearGradient id="riskGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="#22c55e" />
                        <stop offset="45%" stopColor="#a3e635" />
                        <stop offset="65%" stopColor="#facc15" />
                        <stop offset="82%" stopColor="#f97316" />
                        <stop offset="100%" stopColor="#ef4444" />
                      </linearGradient>
                    </defs>

                    <path
                      d="M 45 185 A 135 135 0 0 1 315 185"
                      fill="none"
                      stroke="#263247"
                      strokeWidth="25"
                      strokeLinecap="round"
                    />

                    <path
                      d="M 45 185 A 135 135 0 0 1 315 185"
                      fill="none"
                      stroke="url(#riskGradient)"
                      strokeWidth="25"
                      strokeLinecap="round"
                      pathLength="100"
                      strokeDasharray={`${riskScore} 100`}
                    />

                    <text x="45" y="225" fill="#94a3b8" fontSize="14" textAnchor="middle">0</text>
                    <text x="315" y="225" fill="#94a3b8" fontSize="14" textAnchor="middle">100</text>
                    <text x="180" y="115" fill="#94a3b8" fontSize="14" textAnchor="middle">Average score</text>
                    <text x="180" y="157" fill="#ffffff" fontSize="43" fontWeight="bold" textAnchor="middle">
                      {riskScore}%
                    </text>
                  </svg>

                  <div className="-mt-16 flex justify-center">
                    <span className={`rounded-full px-7 py-2.5 text-sm font-semibold text-white ${riskScore <= 25 ? "bg-green-600" :
                      riskScore <= 50 ? "bg-yellow-600" :
                        riskScore <= 75 ? "bg-orange-500" : "bg-red-600"
                      }`}>
                      {riskScore <= 25 ? "Low" :
                        riskScore <= 50 ? "Moderate" :
                          riskScore <= 75 ? "High" : "Critical"}
                    </span>
                  </div>
                </div>

                <p className="mt-4 text-center text-xs text-slate-500">
                  Calculated from actual saved scan risk scores.
                </p>
              </div>
            </section>

            {/* Seven-day activity */}
            <section className="rounded-2xl border border-white/10 bg-[#101626] p-6">
              <h3 className="font-semibold">Seven-Day Scan Activity</h3>
              <p className="mt-1 text-xs text-slate-500">
                Actual daily scans and detected threats
              </p>

              <div className="mt-5">
                <ResponsiveContainer width="100%" height={250}>
                  <AreaChart data={dailyActivity}>
                    <defs>
                      <linearGradient id="scanArea" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.4} />
                        <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
                      </linearGradient>
                    </defs>

                    <CartesianGrid stroke="#263247" strokeDasharray="3 3" />
                    <XAxis
                      dataKey="label"
                      stroke="#64748b"
                      tick={{ fill: "#94a3b8", fontSize: 12 }}
                    />
                    <YAxis
                      allowDecimals={false}
                      stroke="#64748b"
                      tick={{ fill: "#94a3b8", fontSize: 12 }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#151e30",
                        border: "1px solid #334155",
                        borderRadius: "12px",
                        color: "#fff",
                      }}
                    />
                    <Area
                      type="monotone"
                      dataKey="scans"
                      name="Total scans"
                      stroke="#3b82f6"
                      fill="url(#scanArea)"
                      strokeWidth={2}
                    />
                    <Area
                      type="monotone"
                      dataKey="threats"
                      name="Threats"
                      stroke="#ef4444"
                      fill="none"
                      strokeWidth={2}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </section>

            {/* Recent scans and system status */}
            <section className="grid gap-6 xl:grid-cols-3">

              <div className="rounded-2xl border border-white/10 bg-[#101626] p-6 xl:col-span-2">
                <div className="mb-6 flex items-center justify-between">
                  <div>
                    <h3 className="font-semibold">Recent Scan Activity</h3>
                    <p className="mt-1 text-xs text-slate-500">
                      Latest saved analyses
                    </p>
                  </div>

                  <button
                    onClick={() => setActivePage("history")}
                    className="text-xs font-medium text-blue-400 hover:text-blue-300"
                  >
                    View all
                  </button>
                </div>

                {recentScans.length ? (
                  <div className="space-y-1">
                    {recentScans.map((scan) => {
                      const verdict = (scan.verdict || "UNKNOWN").toUpperCase();

                      return (
                        <div
                          key={scan.scan_id}
                          className="flex flex-col justify-between gap-3 rounded-xl px-3 py-4 transition hover:bg-white/[0.03] sm:flex-row sm:items-center"
                        >
                          <div className="flex min-w-0 items-center gap-3">
                            <div className="rounded-lg bg-white/5 p-2.5 text-slate-400">
                              <ScanSearch size={18} />
                            </div>

                            <div className="min-w-0">
                              <p className="truncate text-sm font-medium">
                                {getScanName(scan)}
                              </p>
                              <p className="mt-1 text-xs text-slate-500">
                                {getScanType(scan)} · {formatDate(scan.created_at)}
                              </p>
                            </div>
                          </div>

                          <span className={`w-fit shrink-0 rounded-lg px-3 py-1.5 text-xs font-semibold ${verdict === "SCAM"
                            ? "bg-red-500/10 text-red-400"
                            : verdict === "SAFE"
                              ? "bg-green-500/10 text-green-400"
                              : "bg-yellow-500/10 text-yellow-400"
                            }`}>
                            {verdict} · {Math.round((scan.overall_risk_score || 0) * 100)}%
                          </span>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="py-10 text-center text-sm text-slate-500">
                    No scans saved yet. Start your first analysis.
                  </div>
                )}
              </div>

              {/* Backend status */}
              <div className="rounded-2xl border border-white/10 bg-[#101626] p-6">
                <h3 className="font-semibold">System Status</h3>
                <p className="mt-1 text-xs text-slate-500">
                  Current service availability
                </p>

                <div className="mt-6 flex items-center justify-between">
                  <span className="text-sm text-slate-300">Backend API</span>

                  <span className={`flex items-center gap-2 text-xs ${backendStatus === "connected"
                    ? "text-green-400"
                    : backendStatus === "disconnected"
                      ? "text-red-400"
                      : "text-yellow-400"
                    }`}>
                    {backendStatus === "connected"
                      ? <CircleCheck size={15} />
                      : <Activity size={15} />}
                    {backendStatus === "connected"
                      ? "Connected"
                      : backendStatus === "disconnected"
                        ? "Unavailable"
                        : "Checking..."}
                  </span>
                </div>

                <div className={`mt-7 rounded-xl border p-4 ${backendStatus === "connected"
                  ? "border-green-500/15 bg-green-500/5"
                  : backendStatus === "disconnected"
                    ? "border-red-500/15 bg-red-500/5"
                    : "border-yellow-500/15 bg-yellow-500/5"
                  }`}>
                  <div className={`flex items-center gap-2 text-sm font-medium ${backendStatus === "connected"
                    ? "text-green-400"
                    : backendStatus === "disconnected"
                      ? "text-red-400"
                      : "text-yellow-400"
                    }`}>
                    <Activity size={17} />
                    {backendStatus === "connected"
                      ? "Backend is operational"
                      : backendStatus === "disconnected"
                        ? "Backend connection failed"
                        : "Checking backend connection"}
                  </div>

                  <p className="mt-2 text-xs leading-5 text-slate-500">
                    {backendStatus === "connected"
                      ? "Successfully communicating with the ScamShield AI backend."
                      : backendStatus === "disconnected"
                        ? "Unable to reach the backend. Check whether the server is running."
                        : "Waiting for a response from the backend API."}
                  </p>
                </div>

                <div className="mt-5 flex items-center gap-2 text-xs text-slate-500">
                  <Clock size={14} /> Live connection status
                </div>
              </div>
            </section>

          </div>
        )}
      </main>
    </div>
  );
}

export default App;

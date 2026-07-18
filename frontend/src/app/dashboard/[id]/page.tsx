"use client";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import { motion, AnimatePresence } from "framer-motion";
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter, ReferenceLine
} from "recharts";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Summary {
  total_rows: number;
  total_columns: number;
  numeric_columns: string[];
  text_columns: string[];
  summary: Record<string, { mean: number; median: number; min: number; max: number; std: number; nulls: number; null_percent: number; }>;
}
interface ChatMessage { role: "user" | "ai"; text: string; }

const fadeUp = {
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -8 },
  transition: { duration: 0.25 }
};

export default function Dashboard() {
  const { id } = useParams<{ id: string }>();
  const [summary, setSummary] = useState<Summary | null>(null);
  const [chartData, setChartData] = useState<any>(null);
  const [xCol, setXCol] = useState("");
  const [yCol, setYCol] = useState("");
  const [chartLoading, setChartLoading] = useState(false);
  const [chartError, setChartError] = useState("");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [tab, setTab] = useState("overview");
  const [mlMode, setMlMode] = useState<"tabular" | "text">("tabular");
  const [qualityData, setQualityData] = useState<any>(null);
  const [qualityLoading, setQualityLoading] = useState(false);
  const [qualityRefreshing, setQualityRefreshing] = useState(false);
  const [qualityTimestamp, setQualityTimestamp] = useState<string>("");
  const [anomalyData, setAnomalyData] = useState<any>(null);
  const [anomalyLoading, setAnomalyLoading] = useState(false);
  const [anomalyError, setAnomalyError] = useState("");
  const [forecastData, setForecastData] = useState<any>(null);
  const [forecastTarget, setForecastTarget] = useState("");
  const [forecastLoading, setForecastLoading] = useState(false);
  const [forecastError, setForecastError] = useState("");
  const [inferenceInputs, setInferenceInputs] = useState<Record<string, string>>({});
  const [inferenceResult, setInferenceResult] = useState<any>(null);
  const [inferenceLoading, setInferenceLoading] = useState(false);
  const [textCol, setTextCol] = useState("");
  const [textTargetCol, setTextTargetCol] = useState("");
  const [textResult, setTextResult] = useState<any>(null);
  const [textLoading, setTextLoading] = useState(false);
  const [textError, setTextError] = useState("");

  useEffect(() => {
    axios.get(API + "/api/analytics/" + id + "/summary").then(res => {
      setSummary(res.data);
      if (res.data.numeric_columns.length >= 1) {
        setYCol(res.data.numeric_columns[0]);
        setForecastTarget(res.data.numeric_columns[res.data.numeric_columns.length - 1]);
      }
      const all = [...res.data.text_columns, ...res.data.numeric_columns];
      if (all.length >= 1) setXCol(all[0]);
      if (res.data.text_columns.length >= 1) {
        setTextCol(res.data.text_columns[0]);
        setTextTargetCol(res.data.text_columns[res.data.text_columns.length - 1]);
      }
    });
  }, [id]);

  async function loadAnomalies() {
    setAnomalyLoading(true);
    setAnomalyError("");
    setAnomalyData(null);
    try {
      const res = await axios.get(API + "/api/analytics/" + id + "/anomalies");
      setAnomalyData(res.data);
    } catch (e: any) {
      setAnomalyError(e?.response?.data?.detail || "Could not run anomaly detection.");
    }
    setAnomalyLoading(false);
  }

  async function loadQuality(isRefresh: boolean = false) {
    if (isRefresh) {
      setQualityRefreshing(true);
    } else {
      setQualityLoading(true);
      setQualityData(null);
    }
    try {
      const res = await axios.get(API + "/api/analytics/" + id + "/quality");
      setQualityData(res.data);
      setQualityTimestamp(new Date().toLocaleTimeString());
    } catch (e) { console.error(e); }
    setQualityLoading(false);
    setQualityRefreshing(false);
  }

  async function loadChart() {
    if (!xCol) return;
    setChartLoading(true); setChartError(""); setChartData(null);
    try {
      const params: any = { x_col: xCol };
      if (yCol) params.y_col = yCol;
      const res = await axios.get(API + "/api/analytics/" + id + "/chart-data", { params });
      setChartData(res.data);
    } catch (e: any) {
      setChartError(e?.response?.data?.detail || "Could not generate chart.");
    }
    setChartLoading(false);
  }

  async function runForecast() {
    if (!forecastTarget) return;
    setForecastLoading(true); setForecastError(""); setForecastData(null); setInferenceResult(null);
    try {
      const res = await axios.get(API + "/api/forecast/" + id + "/forecast", { params: { target_col: forecastTarget } });
      setForecastData(res.data);
      const defaults: Record<string, string> = {};
      (res.data.numeric_input_columns || []).forEach((col: string) => { defaults[col] = ""; });
      setInferenceInputs(defaults);
    } catch (e: any) {
      setForecastError(e?.response?.data?.detail || "Could not run forecast.");
    }
    setForecastLoading(false);
  }

  async function runInference() {
    setInferenceLoading(true); setInferenceResult(null);
    try {
      const res = await axios.post(API + "/api/forecast/" + id + "/predict", {
        target_col: forecastTarget,
        input_values: inferenceInputs
      });
      setInferenceResult(res.data);
    } catch (e: any) {
      setInferenceResult({ error: e?.response?.data?.detail || "Prediction failed." });
    }
    setInferenceLoading(false);
  }

  async function runTextClassification() {
    if (!textCol || !textTargetCol) return;
    setTextLoading(true); setTextError(""); setTextResult(null);
    try {
      const res = await axios.get(API + "/api/forecast/" + id + "/text-classify", { params: { text_col: textCol, target_col: textTargetCol } });
      setTextResult(res.data);
    } catch (e: any) {
      setTextError(e?.response?.data?.detail || "Could not run text classification.");
    }
    setTextLoading(false);
  }

  async function sendMessage() {
    if (!question.trim()) return;
    const userMsg: ChatMessage = { role: "user", text: question };
    const newMessages = [...chatMessages, userMsg];
    setChatMessages(newMessages); setQuestion(""); setChatLoading(true);
    const history = chatMessages.map(m => ({ role: m.role === "user" ? "user" : "model", parts: [{ text: m.text }] }));
    try {
      const res = await axios.post(API + "/api/chat/" + id + "/chat", { question: userMsg.text, chat_history: history });
      setChatMessages([...newMessages, { role: "ai", text: res.data.answer }]);
    } catch {
      setChatMessages([...newMessages, { role: "ai", text: "Error connecting to AI." }]);
    }
    setChatLoading(false);
  }

  const rechartsData = chartData ? chartData.x.map((label: string, i: number) => ({ name: label, value: chartData.y[i] })) : [];

  if (!summary) return (
    <div className="min-h-screen bg-[#0a0a0f] flex items-center justify-center">
      <div className="text-center">
        <div className="w-10 h-10 border-2 border-violet-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
        <p className="text-gray-400 text-sm">Loading dataset...</p>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white">
      <div className="border-b border-white/5 px-6 h-14 flex items-center justify-between sticky top-0 bg-[#0a0a0f]/90 backdrop-blur-md z-10">
        <div className="flex items-center gap-3">
          <a href="/" className="text-gray-400 text-sm hover:text-white transition-colors">InsightFlow</a>
          <span className="text-gray-700">/</span>
          <span className="text-sm font-medium">Dataset #{id}</span>
        </div>
        <div className="flex gap-1 bg-white/5 rounded-xl p-1">
          {["overview", "quality", "anomalies", "charts", "forecast", "ai"].map(t => (
            <button key={t} onClick={() => setTab(t)}
              className={"relative px-4 py-1.5 rounded-lg text-sm font-medium transition-all " + (tab === t ? "text-white" : "text-gray-400 hover:text-white")}>
              {tab === t && (
                <motion.div layoutId="tab-pill" className="absolute inset-0 bg-violet-600 rounded-lg" style={{ zIndex: -1 }} transition={{ type: "spring", bounce: 0.2, duration: 0.4 }} />
              )}
              {t === "ai" ? "Ask AI" : t === "forecast" ? "Forecast" : t === "quality" ? "Data Quality" : t === "anomalies" ? "Anomalies" : t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-8">
        <AnimatePresence mode="wait">

          {tab === "overview" && (
            <motion.div key="overview" {...fadeUp} className="space-y-6">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  { label: "Total Rows", value: summary.total_rows.toLocaleString(), color: "violet" },
                  { label: "Total Columns", value: summary.total_columns, color: "blue" },
                  { label: "Numeric Columns", value: summary.numeric_columns.length, color: "emerald" },
                  { label: "Text Columns", value: summary.text_columns.length, color: "amber" },
                ].map((card, i) => (
                  <motion.div key={card.label} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.07 }}
                    className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-5 hover:border-white/10 transition-colors">
                    <p className="text-xs text-gray-500 mb-2 uppercase tracking-wider">{card.label}</p>
                    <p className="text-3xl font-semibold">{card.value}</p>
                  </motion.div>
                ))}
              </div>
              <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl overflow-hidden">
                <div className="px-6 py-4 border-b border-white/[0.06]">
                  <h2 className="text-sm font-semibold">Column Statistics</h2>
                  <p className="text-xs text-gray-500 mt-0.5">Full-dataset summary for all numeric columns</p>
                </div>
                {Object.keys(summary.summary).length === 0 ? (
                  <div className="px-6 py-12 text-center">
                    <p className="text-gray-500 text-sm">No numeric columns. Try Charts or Forecast tabs.</p>
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-white/[0.06]">
                          {["Column", "Mean", "Median", "Min", "Max", "Std Dev", "Nulls"].map(h => (
                            <th key={h} className="text-left px-6 py-3 text-xs text-gray-500 uppercase tracking-wider">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(summary.summary).map(([col, stats]) => (
                          <tr key={col} className="border-b border-white/[0.03] hover:bg-white/[0.02] transition-colors">
                            <td className="px-6 py-3 text-violet-300 font-medium">{col}</td>
                            <td className="px-6 py-3 text-gray-300">{stats.mean}</td>
                            <td className="px-6 py-3 text-gray-300">{stats.median}</td>
                            <td className="px-6 py-3 text-gray-300">{stats.min}</td>
                            <td className="px-6 py-3 text-gray-300">{stats.max}</td>
                            <td className="px-6 py-3 text-gray-300">{stats.std}</td>
                            <td className="px-6 py-3">
                              <span className={"text-xs px-2 py-0.5 rounded-full " + (stats.nulls > 0 ? "bg-red-500/10 text-red-400" : "bg-emerald-500/10 text-emerald-400")}>
                                {stats.nulls} ({stats.null_percent}%)
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </motion.div>
          )}

          {tab === "quality" && (
            <motion.div key="quality" {...fadeUp} className="space-y-6">
              {!qualityData ? (
                <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-8 text-center">
                  <p className="text-gray-400 text-sm mb-4">Run a quality check to identify issues before training your ML model</p>
                  {qualityTimestamp && <p className="text-xs text-gray-600">Last checked: {qualityTimestamp}</p>}
                  <button onClick={() => loadQuality(true)} disabled={qualityLoading}
                    className="bg-violet-600 hover:bg-violet-500 disabled:opacity-40 text-white px-6 py-2.5 rounded-lg text-sm font-medium transition-all active:scale-95 flex items-center gap-2 mx-auto">
                    {qualityLoading ? (
                      <><span className="w-3.5 h-3.5 border border-white/30 border-t-white rounded-full animate-spin"></span>Analysing...</>
                    ) : "Run Data Quality Check"}
                  </button>
                </div>
              ) : (
                <>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-5 md:col-span-1">
                      <p className="text-xs text-gray-500 mb-2 uppercase tracking-wider">Quality Score</p>
                      <p className={"text-4xl font-bold " + (qualityData.grade_color === "emerald" ? "text-emerald-400" : qualityData.grade_color === "blue" ? "text-blue-400" : qualityData.grade_color === "amber" ? "text-amber-400" : "text-red-400")}>
                        {qualityData.score}/100
                      </p>
                      <p className="text-xs text-gray-500 mt-1">{qualityData.grade}</p>
                    </div>
                    <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-5">
                      <p className="text-xs text-gray-500 mb-2 uppercase tracking-wider">Duplicate Rows</p>
                      <p className={"text-3xl font-semibold " + (qualityData.duplicate_count > 0 ? "text-amber-400" : "text-emerald-400")}>{qualityData.duplicate_count}</p>
                      <p className="text-xs text-gray-600 mt-1">{qualityData.duplicate_percent}% of dataset</p>
                    </div>
                    <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-5">
                      <p className="text-xs text-gray-500 mb-2 uppercase tracking-wider">Constant Columns</p>
                      <p className={"text-3xl font-semibold " + (qualityData.constant_columns.length > 0 ? "text-red-400" : "text-emerald-400")}>{qualityData.constant_columns.length}</p>
                      <p className="text-xs text-gray-600 mt-1">zero variance</p>
                    </div>
                    <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-5">
                      <p className="text-xs text-gray-500 mb-2 uppercase tracking-wider">ML Ready</p>
                      <p className={"text-3xl font-semibold " + (qualityData.ml_ready ? "text-emerald-400" : "text-red-400")}>{qualityData.ml_ready ? "Yes" : "No"}</p>
                    </div>
                  </div>

                  {qualityData.issues.length > 0 && (
                    <div className="bg-red-500/5 border border-red-500/20 rounded-2xl p-6">
                      <h3 className="text-sm font-semibold text-red-400 mb-3">Issues — fix these before training</h3>
                      <div className="space-y-2">
                        {qualityData.issues.map((issue: string, i: number) => (
                          <div key={i} className="flex items-start gap-2 text-xs text-gray-300">
                            <span className="text-red-400 mt-0.5">x</span>
                            <span>{issue}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {qualityData.warnings.length > 0 && (
                    <div className="bg-amber-500/5 border border-amber-500/20 rounded-2xl p-6">
                      <h3 className="text-sm font-semibold text-amber-400 mb-3">Warnings — worth knowing</h3>
                      <div className="space-y-2">
                        {qualityData.warnings.map((w: string, i: number) => (
                          <div key={i} className="flex items-start gap-2 text-xs text-gray-300">
                            <span className="text-amber-400 mt-0.5">!</span>
                            <span>{w}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {qualityData.info.length > 0 && (
                    <div className="bg-blue-500/5 border border-blue-500/20 rounded-2xl p-6">
                      <h3 className="text-sm font-semibold text-blue-400 mb-3">Info</h3>
                      <div className="space-y-2">
                        {qualityData.info.map((item: string, i: number) => (
                          <div key={i} className="flex items-start gap-2 text-xs text-gray-300">
                            <span className="text-blue-400 mt-0.5">i</span>
                            <span>{item}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {qualityData.null_report.length > 0 && (
                    <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl overflow-hidden">
                      <div className="px-6 py-4 border-b border-white/[0.06]">
                        <h3 className="text-sm font-semibold">Missing Values by Column</h3>
                      </div>
                      <div className="p-6 space-y-3">
                        {qualityData.null_report.map((col: any) => (
                          <div key={col.column} className="flex items-center gap-3">
                            <span className="text-xs text-gray-400 w-24 truncate">{col.column}</span>
                            <div className="flex-1 bg-white/5 rounded-full h-2 overflow-hidden">
                              <motion.div initial={{ width: 0 }} animate={{ width: col.null_percent + "%" }} transition={{ duration: 0.5 }}
                                className={"h-full rounded-full " + (col.severity === "high" ? "bg-red-500" : col.severity === "medium" ? "bg-amber-500" : "bg-blue-500")} />
                            </div>
                            <span className="text-xs text-gray-500 w-16 text-right">{col.null_percent}% null</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {qualityData.correlation_issues.length > 0 && (
                    <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-6">
                      <h3 className="text-sm font-semibold mb-1">Highly Correlated Column Pairs</h3>
                      <p className="text-xs text-gray-500 mb-4">These columns carry nearly identical information — one could be removed</p>
                      <div className="space-y-2">
                        {qualityData.correlation_issues.map((pair: any, i: number) => (
                          <div key={i} className="flex items-center justify-between bg-white/[0.02] rounded-lg px-4 py-2">
                            <span className="text-xs text-gray-300">{pair.col1} + {pair.col2}</span>
                            <span className="text-xs text-amber-400">{(pair.correlation * 100).toFixed(1)}% correlated</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

<button onClick={() => loadQuality(true)} className="text-xs text-violet-400 hover:text-violet-300 transition-colors flex items-center gap-1.5">
                    {qualityRefreshing ? (
                      <><span className="w-3 h-3 border border-violet-400 border-t-transparent rounded-full animate-spin"></span>Refreshing...</>
                    ) : "Re-run quality check"}
                  </button>
                </>
              )}
            </motion.div>
          )}

          {tab === "anomalies" && (
            <motion.div key="anomalies" {...fadeUp} className="space-y-6">
              {!anomalyData && !anomalyLoading && (
                <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-8 text-center">
                  <p className="text-gray-400 text-sm mb-2">Detect statistically unusual rows in your dataset</p>
                  <p className="text-gray-600 text-xs mb-6">Uses IQR method per column + Isolation Forest for global anomaly detection</p>
                  <button onClick={loadAnomalies}
                    className="bg-violet-600 hover:bg-violet-500 text-white px-6 py-2.5 rounded-lg text-sm font-medium transition-all active:scale-95 mx-auto flex items-center gap-2">
                    Run Anomaly Detection
                  </button>
                </div>
              )}

              {anomalyLoading && (
                <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-8 text-center">
                  <div className="w-8 h-8 border-2 border-violet-500 border-t-transparent rounded-full animate-spin mx-auto mb-3"></div>
                  <p className="text-gray-400 text-sm">Scanning {summary.total_rows.toLocaleString()} rows for anomalies...</p>
                </div>
              )}

              {anomalyError && (
                <div className="bg-red-500/5 border border-red-500/20 rounded-2xl p-6 text-center">
                  <p className="text-red-400 text-sm font-medium mb-1">Detection failed</p>
                  <p className="text-gray-500 text-xs">{anomalyError}</p>
                </div>
              )}

              {anomalyData && (
                <>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-5">
                      <p className="text-xs text-gray-500 mb-2 uppercase tracking-wider">Anomalies Found</p>
                      <p className={"text-3xl font-semibold " + (anomalyData.total_anomalies > 0 ? "text-amber-400" : "text-emerald-400")}>{anomalyData.total_anomalies}</p>
                      <p className="text-xs text-gray-600 mt-1">{anomalyData.anomaly_percent}% of dataset</p>
                    </div>
                    <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-5">
                      <p className="text-xs text-gray-500 mb-2 uppercase tracking-wider">Total Rows</p>
                      <p className="text-3xl font-semibold">{anomalyData.total_rows.toLocaleString()}</p>
                    </div>
                    <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-5">
                      <p className="text-xs text-gray-500 mb-2 uppercase tracking-wider">Columns Analyzed</p>
                      <p className="text-3xl font-semibold">{anomalyData.numeric_columns_analyzed.length}</p>
                    </div>
                    <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-5">
                      <p className="text-xs text-gray-500 mb-2 uppercase tracking-wider">Method</p>
                      <p className="text-sm font-medium text-violet-300 mt-2">IQR + Isolation Forest</p>
                    </div>
                  </div>

                  <div className="bg-violet-500/5 border border-violet-500/20 rounded-xl px-4 py-3">
                    <p className="text-xs text-violet-300 font-medium mb-1">Summary</p>
                    <p className="text-xs text-gray-400">{anomalyData.summary}</p>
                  </div>

                  {anomalyData.column_anomalies.length > 0 && (
                    <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-6">
                      <h3 className="text-sm font-semibold mb-1">Outliers by Column (IQR Method)</h3>
                      <p className="text-xs text-gray-500 mb-5">Values outside 1.5x the interquartile range are flagged as outliers</p>
                      <div className="space-y-4">
                        {anomalyData.column_anomalies.map((col: any) => (
                          <div key={col.column}>
                            <div className="flex items-center justify-between mb-1.5">
                              <div className="flex items-center gap-2">
                                <span className="text-xs text-gray-300 font-medium">{col.column}</span>
                                <span className={"text-xs px-2 py-0.5 rounded-full " + (col.severity === "high" ? "bg-red-500/10 text-red-400" : col.severity === "medium" ? "bg-amber-500/10 text-amber-400" : "bg-blue-500/10 text-blue-400")}>
                                  {col.outlier_count} outliers ({col.outlier_percent}%)
                                </span>
                              </div>
                              <span className="text-xs text-gray-600">normal: {col.lower_bound} to {col.upper_bound}</span>
                            </div>
                            <div className="flex-1 bg-white/5 rounded-full h-1.5 overflow-hidden">
                              <motion.div initial={{ width: 0 }} animate={{ width: col.outlier_percent + "%" }} transition={{ duration: 0.5 }}
                                className={"h-full rounded-full " + (col.severity === "high" ? "bg-red-500" : col.severity === "medium" ? "bg-amber-500" : "bg-blue-500")} />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {anomalyData.anomaly_rows.length > 0 && (
                    <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl overflow-hidden">
                      <div className="px-6 py-4 border-b border-white/[0.06]">
                        <h3 className="text-sm font-semibold">Most Anomalous Rows</h3>
                        <p className="text-xs text-gray-500 mt-0.5">Rows with the most unusual combination of values, detected by Isolation Forest</p>
                      </div>
                      <div className="divide-y divide-white/[0.04]">
                        {anomalyData.anomaly_rows.slice(0, 10).map((row: any, i: number) => (
                          <motion.div key={i} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.04 }}
                            className="px-6 py-4">
                            <div className="flex items-start justify-between mb-2">
                              <span className="text-xs text-gray-500">Row #{row.row_index}</span>
                              <span className="text-xs text-amber-400">anomaly score: {row.anomaly_score}</span>
                            </div>
                            <div className="space-y-1">
                              {row.reasons.map((reason: string, j: number) => (
                                <p key={j} className="text-xs text-gray-400 flex items-start gap-1.5">
                                  <span className="text-amber-500 mt-0.5">!</span>
                                  {reason}
                                </p>
                              ))}
                            </div>
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  )}

                  <button onClick={loadAnomalies} className="text-xs text-violet-400 hover:text-violet-300 transition-colors">
                    Re-run detection
                  </button>
                </>
              )}
            </motion.div>
          )}

          {tab === "charts" && (
            <motion.div key="charts" {...fadeUp} className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-6">
              <h2 className="text-sm font-semibold mb-1">Chart Builder</h2>
              <p className="text-xs text-gray-500 mb-5">Select columns to visualize relationships in your data</p>
              <div className="flex gap-3 flex-wrap mb-6">
                <div>
                  <label className="block text-xs text-gray-500 mb-1.5 uppercase tracking-wider">X Axis</label>
                  <select value={xCol} onChange={e => setXCol(e.target.value)} className="bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-violet-500 transition-colors">
                    {[...summary.text_columns, ...summary.numeric_columns].map(c => (
                      <option key={c} value={c} className="bg-gray-900">{c}</option>
                    ))}
                  </select>
                </div>
                {summary.numeric_columns.length > 0 && (
                  <div>
                    <label className="block text-xs text-gray-500 mb-1.5 uppercase tracking-wider">Y Axis</label>
                    <select value={yCol} onChange={e => setYCol(e.target.value)} className="bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-violet-500 transition-colors">
                      {summary.numeric_columns.map(c => (
                        <option key={c} value={c} className="bg-gray-900">{c}</option>
                      ))}
                    </select>
                  </div>
                )}
                <div className="flex items-end">
                  <button onClick={loadChart} disabled={!xCol || chartLoading}
                    className="bg-violet-600 hover:bg-violet-500 disabled:opacity-40 text-white px-5 py-2 rounded-lg text-sm font-medium transition-all active:scale-95">
                    {chartLoading ? "Loading..." : "Generate Chart"}
                  </button>
                </div>
              </div>
              <AnimatePresence mode="wait">
                {chartData && rechartsData.length > 0 ? (
                  <motion.div key="chart" {...fadeUp}>
                    <p className="text-xs text-gray-500 mb-4">{chartData.y_label} by {chartData.x_label}</p>
                    <ResponsiveContainer width="100%" height={350}>
                      {chartData.is_timeseries ? (
                        <LineChart data={rechartsData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                          <XAxis dataKey="name" tick={{ fontSize: 10, fill: "#6b7280" }} />
                          <YAxis tick={{ fontSize: 10, fill: "#6b7280" }} />
                          <Tooltip contentStyle={{ backgroundColor: "#13131a", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "8px", color: "#fff" }} />
                          <Line type="monotone" dataKey="value" stroke="#7c3aed" strokeWidth={2} dot={false} />
                        </LineChart>
                      ) : (
                        <BarChart data={rechartsData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                          <XAxis dataKey="name" tick={{ fontSize: 10, fill: "#6b7280" }} />
                          <YAxis tick={{ fontSize: 10, fill: "#6b7280" }} />
                          <Tooltip contentStyle={{ backgroundColor: "#13131a", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "8px", color: "#fff" }} />
                          <Bar dataKey="value" fill="#7c3aed" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      )}
                    </ResponsiveContainer>
                  </motion.div>
                ) : (
                  <motion.div key="empty" {...fadeUp} className="h-64 flex items-center justify-center border border-dashed border-white/10 rounded-xl">
                    {chartError ? (
                      <div className="text-center px-6">
                        <p className="text-3xl mb-2">!</p>
                        <p className="text-red-400 text-sm font-medium">Cannot chart this column</p>
                        <p className="text-gray-500 text-xs mt-1">{chartError}</p>
                      </div>
                    ) : (
                      <p className="text-gray-600 text-sm">Select columns and click Generate Chart</p>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          )}

          {tab === "forecast" && (
            <motion.div key="forecast" {...fadeUp} className="space-y-6">
              <div className="flex gap-1 bg-white/5 rounded-xl p-1 w-fit">
                {(["tabular", "text"] as const).map(mode => (
                  <button key={mode} onClick={() => setMlMode(mode)}
                    className={"relative px-4 py-1.5 rounded-lg text-sm font-medium transition-all " + (mlMode === mode ? "text-white" : "text-gray-400 hover:text-white")}>
                    {mlMode === mode && (
                      <motion.div layoutId="ml-pill" className="absolute inset-0 bg-violet-600 rounded-lg" style={{ zIndex: -1 }} transition={{ type: "spring", bounce: 0.2, duration: 0.4 }} />
                    )}
                    {mode === "tabular" ? "Tabular ML" : "Text Classification"}
                  </button>
                ))}
              </div>

              <AnimatePresence mode="wait">
                {mlMode === "tabular" && (
                  <motion.div key="tabular" {...fadeUp} className="space-y-6">
                    <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-6">
                      <div className="flex items-start justify-between mb-4">
                        <div>
                          <h2 className="text-sm font-semibold">ML Forecast</h2>
                          <p className="text-xs text-gray-500 mt-0.5">Train XGBoost to predict any column, then use the model for real predictions</p>
                        </div>
                        {forecastData && (
                          <span className="text-xs bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-3 py-1 rounded-full">
                            Model trained
                          </span>
                        )}
                      </div>
                      <div className="flex gap-3 flex-wrap">
                        <div>
                          <label className="block text-xs text-gray-500 mb-1.5 uppercase tracking-wider">Target Column</label>
                          <select value={forecastTarget} onChange={e => { setForecastTarget(e.target.value); setForecastData(null); setInferenceResult(null); }}
                            className="bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-violet-500 transition-colors">
                            {summary.numeric_columns.map(c => (
                              <option key={c} value={c} className="bg-gray-900">{c}</option>
                            ))}
                          </select>
                        </div>
                        <div className="flex items-end">
                          <button onClick={runForecast} disabled={!forecastTarget || forecastLoading}
                            className="bg-violet-600 hover:bg-violet-500 disabled:opacity-40 text-white px-5 py-2 rounded-lg text-sm font-medium transition-all active:scale-95 flex items-center gap-2">
                            {forecastLoading ? (
                              <>
                                <span className="w-3.5 h-3.5 border border-white/30 border-t-white rounded-full animate-spin"></span>
                                Training...
                              </>
                            ) : "Run Forecast"}
                          </button>
                        </div>
                      </div>
                    </div>

                    {forecastError && (
                      <motion.div {...fadeUp} className="bg-red-500/5 border border-red-500/20 rounded-2xl p-6 text-center">
                        <p className="text-red-400 text-sm font-medium mb-1">Forecast failed</p>
                        <p className="text-gray-500 text-xs">{forecastError}</p>
                      </motion.div>
                    )}

                    {forecastData && (
                      <motion.div {...fadeUp} className="space-y-6">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          {forecastData.is_classification ? (
                            <>
                              <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-5">
                                <p className="text-xs text-gray-500 mb-2 uppercase tracking-wider">Accuracy</p>
                                <p className="text-3xl font-semibold text-emerald-400">{(forecastData.accuracy * 100).toFixed(1)}%</p>
                              </div>
                              <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-5">
                                <p className="text-xs text-gray-500 mb-2 uppercase tracking-wider">F1 Score</p>
                                <p className="text-3xl font-semibold">{forecastData.f1_score}</p>
                              </div>
                            </>
                          ) : (
                            <>
                              <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-5">
                                <p className="text-xs text-gray-500 mb-2 uppercase tracking-wider">R2 Score</p>
                                <p className="text-3xl font-semibold text-emerald-400">{forecastData.r2_score}</p>
                                <p className="text-xs text-gray-600 mt-1">Explains {Math.round(forecastData.r2_score * 100)}% of variance</p>
                              </div>
                              <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-5">
                                <p className="text-xs text-gray-500 mb-2 uppercase tracking-wider">Mean Abs Error</p>
                                <p className="text-3xl font-semibold">{forecastData.mae}</p>
                              </div>
                            </>
                          )}
                          <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-5">
                            <p className="text-xs text-gray-500 mb-2 uppercase tracking-wider">CV Score</p>
                            <p className="text-3xl font-semibold">{forecastData.cv_mean}</p>
                            <p className="text-xs text-gray-600 mt-1">+/- {forecastData.cv_std} (5-fold)</p>
                          </div>
                          <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-5">
                            <p className="text-xs text-gray-500 mb-2 uppercase tracking-wider">Test Rows</p>
                            <p className="text-3xl font-semibold">{forecastData.test_rows}</p>
                          </div>
                        </div>

                        {!forecastData.is_classification && forecastData.actual_vs_predicted && (
                          <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-6">
                            <h3 className="text-sm font-semibold mb-1">Actual vs Predicted</h3>
                            <p className="text-xs text-gray-500 mb-1">Each dot is one test row. The diagonal line = perfect prediction. Points close to it = accurate model.</p>
                            <div className="flex gap-4 mb-5">
                              <span className="text-xs text-violet-400">Purple dots = model predictions</span>
                              <span className="text-xs text-gray-500">White line = perfect accuracy</span>
                            </div>
                            <ResponsiveContainer width="100%" height={320}>
                              <ScatterChart>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis type="number" dataKey="actual" name="Actual" tick={{ fontSize: 10, fill: "#6b7280" }} label={{ value: "Actual value", position: "insideBottom", offset: -5, fill: "#6b7280", fontSize: 11 }} />
                                <YAxis type="number" dataKey="predicted" name="Predicted" tick={{ fontSize: 10, fill: "#6b7280" }} label={{ value: "Predicted", angle: -90, position: "insideLeft", fill: "#6b7280", fontSize: 11 }} />
                                <Tooltip contentStyle={{ backgroundColor: "#13131a", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "8px", color: "#fff" }}
                                  cursor={{ strokeDasharray: "3 3" }} /> 
                                <ReferenceLine segment={[
                                  { x: Math.min(...forecastData.actual_vs_predicted.map((d: any) => d.actual)), y: Math.min(...forecastData.actual_vs_predicted.map((d: any) => d.actual)) },
                                  { x: Math.max(...forecastData.actual_vs_predicted.map((d: any) => d.actual)), y: Math.max(...forecastData.actual_vs_predicted.map((d: any) => d.actual)) }
                                ]} stroke="rgba(255,255,255,0.2)" strokeDasharray="4 4" />
                                <Scatter data={forecastData.actual_vs_predicted} fill="#7c3aed" opacity={0.8} />
                              </ScatterChart>
                            </ResponsiveContainer>
                          </div>
                        )}

                        <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-6">
                          <h3 className="text-sm font-semibold mb-1">Feature Importance (SHAP)</h3>
                          <p className="text-xs text-gray-500 mb-5">Which columns influenced the prediction most</p>
                          <ResponsiveContainer width="100%" height={Math.max(280, forecastData.shap_importance.length * 32)}>
                            <BarChart data={forecastData.shap_importance} layout="vertical" margin={{ left: 10 }}>
                              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                              <XAxis type="number" tick={{ fontSize: 10, fill: "#6b7280" }} />
                              <YAxis type="category" dataKey="feature" tick={{ fontSize: 11, fill: "#9ca3af" }} width={80} />
                              <Tooltip contentStyle={{ backgroundColor: "#13131a", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "8px", color: "#fff" }} />
                              <Bar dataKey="impact" fill="#7c3aed" radius={[0, 4, 4, 0]} />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>

                        <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-6">
                          <h3 className="text-sm font-semibold mb-1">Make a Prediction</h3>
                          <p className="text-xs text-gray-500 mb-5">Enter values for each feature column — leave blank to use the dataset average</p>
                          <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-5">
                            {(forecastData.numeric_input_columns || []).map((col: string) => (
                              <div key={col}>
                                <label className="block text-xs text-gray-500 mb-1">{col}</label>
                                <input
                                  type="number"
                                  placeholder={"avg: " + (summary.summary[col]?.mean ?? "?")}
                                  value={inferenceInputs[col] || ""}
                                  onChange={e => setInferenceInputs(prev => ({ ...prev, [col]: e.target.value }))}
                                  className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-violet-500 transition-colors"
                                />
                              </div>
                            ))}
                          </div>
                          <button onClick={runInference} disabled={inferenceLoading}
                            className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white px-6 py-2.5 rounded-lg text-sm font-medium transition-all active:scale-95 flex items-center gap-2">
                            {inferenceLoading ? (
                              <>
                                <span className="w-3.5 h-3.5 border border-white/30 border-t-white rounded-full animate-spin"></span>
                                Predicting...
                              </>
                            ) : "Predict"}
                          </button>

                          <AnimatePresence>
                            {inferenceResult && (
                              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="mt-5">
                                {inferenceResult.error ? (
                                  <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4">
                                    <p className="text-red-400 text-sm">{inferenceResult.error}</p>
                                  </div>
                                ) : (
                                  <div className="bg-emerald-500/5 border border-emerald-500/20 rounded-xl p-5">
                                    <p className="text-xs text-gray-500 mb-2 uppercase tracking-wider">Predicted {forecastData.target_column}</p>
                                    <p className="text-4xl font-bold text-emerald-400">{inferenceResult.prediction}</p>
                                    {inferenceResult.confidence_interval && (
                                      <div className="mt-4">
                                        <p className="text-xs text-gray-500 uppercase tracking-wider mb-3">Confidence Range (based on test MAE)</p>
                                        <div className="flex items-center gap-3 mb-2">
                                          <span className="text-sm text-gray-400">{inferenceResult.confidence_interval.low}</span>
                                          <div className="flex-1 relative h-6 flex items-center">
                                            <div className="w-full bg-white/5 rounded-full h-1.5"></div>
                                            <div className="absolute inset-x-0 flex items-center justify-center">
                                              <div className="w-3 h-3 bg-emerald-400 rounded-full border-2 border-[#0a0a0f]"></div>
                                            </div>
                                          </div>
                                          <span className="text-sm text-gray-400">{inferenceResult.confidence_interval.high}</span>
                                        </div>
                                        <p className="text-xs text-gray-600 mt-2">{inferenceResult.confidence_interval.explanation}</p>
                                        <p className="text-xs text-amber-500/70 mt-1">This range covers roughly 68% of similar predictions based on test performance.</p>
                                      </div>
                                    )}
                                    {inferenceResult.probabilities && (
                                      <div className="mt-4 space-y-2">
                                        <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">Class Probabilities</p>
                                        {Object.entries(inferenceResult.probabilities).map(([cls, prob]: [string, any]) => (
                                          <div key={cls} className="flex items-center gap-3">
                                            <span className="text-xs text-gray-400 w-20">{cls}</span>
                                            <div className="flex-1 bg-white/5 rounded-full h-2 overflow-hidden">
                                              <motion.div initial={{ width: 0 }} animate={{ width: prob + "%" }} transition={{ duration: 0.6, ease: "easeOut" }}
                                                className="bg-violet-500 h-full rounded-full" />
                                            </div>
                                            <span className="text-xs text-gray-300 w-12 text-right">{prob}%</span>
                                          </div>
                                        ))}
                                      </div>
                                    )}
                                  </div>
                                )}
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </div>
                      </motion.div>
                    )}
                  </motion.div>
                )}

                {mlMode === "text" && (
                  <motion.div key="text" {...fadeUp} className="space-y-6">
                    <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-6">
                      <h2 className="text-sm font-semibold mb-1">Text Classification</h2>
                      <p className="text-xs text-gray-500 mb-5">Classify free text using TF-IDF + Logistic Regression — works on any text + category dataset</p>
                      <div className="flex gap-3 flex-wrap mb-2">
                        <div>
                          <label className="block text-xs text-gray-500 mb-1.5 uppercase tracking-wider">Text Column</label>
                          <select value={textCol} onChange={e => setTextCol(e.target.value)} className="bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-violet-500">
                            <option value="" className="bg-gray-900">Select column</option>
                            {summary.text_columns.map(c => (<option key={c} value={c} className="bg-gray-900">{c}</option>))}
                          </select>
                        </div>
                        <div>
                          <label className="block text-xs text-gray-500 mb-1.5 uppercase tracking-wider">Target Column</label>
                          <select value={textTargetCol} onChange={e => setTextTargetCol(e.target.value)} className="bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-violet-500">
                            <option value="" className="bg-gray-900">Select column</option>
                            {summary.text_columns.map(c => (<option key={c} value={c} className="bg-gray-900">{c}</option>))}
                          </select>
                        </div>
                        <div className="flex items-end">
                          <button onClick={runTextClassification} disabled={!textCol || !textTargetCol || textLoading}
                            className="bg-violet-600 hover:bg-violet-500 disabled:opacity-40 text-white px-5 py-2 rounded-lg text-sm font-medium transition-all active:scale-95 flex items-center gap-2">
                            {textLoading ? (
                              <>
                                <span className="w-3.5 h-3.5 border border-white/30 border-t-white rounded-full animate-spin"></span>
                                Training (10-20s)...
                              </>
                            ) : "Run Classification"}
                          </button>
                        </div>
                      </div>
                      <p className="text-xs text-gray-600 mt-2">For IMDB: select "review" as text column and "sentiment" as target</p>
                    </div>

                    {textError && (
                      <div className="bg-red-500/5 border border-red-500/20 rounded-2xl p-6 text-center">
                        <p className="text-red-400 text-sm font-medium mb-1">Classification failed</p>
                        <p className="text-gray-500 text-xs">{textError}</p>
                      </div>
                    )}

                    {textResult && (
                      <motion.div {...fadeUp} className="space-y-6">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          {[
                            { label: "Accuracy", value: (textResult.accuracy * 100).toFixed(1) + "%", green: true },
                            { label: "F1 Score", value: textResult.f1_score, green: false },
                            { label: "Vocabulary", value: textResult.vocabulary_size.toLocaleString(), green: false },
                            { label: "Test Rows", value: textResult.test_rows, green: false },
                          ].map(card => (
                            <div key={card.label} className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-5">
                              <p className="text-xs text-gray-500 mb-2 uppercase tracking-wider">{card.label}</p>
                              <p className={"text-3xl font-semibold " + (card.green ? "text-emerald-400" : "")}>{card.value}</p>
                            </div>
                          ))}
                        </div>

                        <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-6">
                          <h3 className="text-sm font-semibold mb-1">Most Influential Words Per Class</h3>
                          <p className="text-xs text-gray-500 mb-5">Words that most strongly push the model toward each class</p>
                          <div className="grid md:grid-cols-2 gap-8">
                            {Object.entries(textResult.top_words_by_class).map(([className, words]: [string, any]) => (
                              <div key={className}>
                                <p className="text-sm font-medium text-violet-300 mb-4 capitalize">{className}</p>
                                <div className="space-y-2">
                                  {words.slice(0, 10).map((w: any, i: number) => (
                                    <motion.div key={i} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.04 }}
                                      className="flex items-center gap-3">
                                      <span className="text-xs text-gray-400 w-24 truncate">{w.word}</span>
                                      <div className="flex-1 bg-white/5 rounded-full h-1.5 overflow-hidden">
                                        <motion.div initial={{ width: 0 }} animate={{ width: Math.min(100, (w.weight / words[0].weight) * 100) + "%" }}
                                          transition={{ duration: 0.5, delay: i * 0.04 }}
                                          className="bg-violet-500 h-full rounded-full" />
                                      </div>
                                      <span className="text-xs text-gray-500 w-12 text-right">{w.weight}</span>
                                    </motion.div>
                                  ))}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>

                        <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-6">
                          <h3 className="text-sm font-semibold mb-1">Sample Predictions</h3>
                          <p className="text-xs text-gray-500 mb-5">Real examples the model classified</p>
                          <div className="space-y-2">
                            {textResult.predictions_sample.map((p: any, i: number) => (
                              <motion.div key={i} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.03 }}
                                className={"rounded-xl px-4 py-3 border text-xs " + (p.correct ? "bg-emerald-500/5 border-emerald-500/20" : "bg-red-500/5 border-red-500/20")}>
                                <p className="text-gray-400 mb-2 leading-relaxed">{p.text_preview}</p>
                                <div className="flex gap-4">
                                  <span className="text-gray-500">Actual: <span className="text-gray-300">{p.actual}</span></span>
                                  <span className="text-gray-500">Predicted: <span className={p.correct ? "text-emerald-400" : "text-red-400"}>{p.predicted}</span></span>
                                </div>
                              </motion.div>
                            ))}
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          )}

          {tab === "ai" && (
            <motion.div key="ai" {...fadeUp} className="bg-white/[0.03] border border-white/[0.06] rounded-2xl overflow-hidden">
              <div className="px-6 py-4 border-b border-white/[0.06]">
                <h2 className="text-sm font-semibold">Ask AI About This Data</h2>
                <p className="text-xs text-gray-500 mt-0.5">Powered by Llama 3.3 via Groq — full dataset context</p>
              </div>
              <div className="h-[500px] overflow-y-auto p-6 space-y-4">
                {chatMessages.length === 0 && (
                  <div className="space-y-2">
                    <p className="text-xs text-gray-600 uppercase tracking-wider mb-4">Suggested questions</p>
                    {["What are the top insights from this data?", "Which column has the most nulls?", "Are there any outliers I should know about?", "What is the distribution of values in this dataset?"].map(q => (
                      <button key={q} onClick={() => setQuestion(q)}
                        className="block w-full text-left text-sm text-gray-400 hover:text-white bg-white/[0.02] hover:bg-white/[0.05] border border-white/[0.05] hover:border-white/10 rounded-xl px-4 py-2.5 transition-all">
                        {q}
                      </button>
                    ))}
                  </div>
                )}
                {chatMessages.map((msg, i) => (
                  <motion.div key={i} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                    className={"flex " + (msg.role === "user" ? "justify-end" : "justify-start")}>
                    <div className={"max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed " + (msg.role === "user" ? "bg-violet-600 text-white rounded-br-sm" : "bg-white/[0.05] text-gray-200 border border-white/[0.06] rounded-bl-sm")}>
                      {msg.role === "ai" ? (
                        <div className="prose prose-invert prose-sm max-w-none">
                          <ReactMarkdown>{msg.text}</ReactMarkdown>
                        </div>
                      ) : msg.text}
                    </div>
                  </motion.div>
                ))}
                {chatLoading && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex justify-start">
                    <div className="bg-white/[0.05] border border-white/[0.06] rounded-2xl rounded-bl-sm px-4 py-3">
                      <div className="flex gap-1.5 items-center">
                        {[0, 1, 2].map(i => (
                          <motion.span key={i} className="w-1.5 h-1.5 bg-gray-500 rounded-full"
                            animate={{ y: [0, -4, 0] }} transition={{ duration: 0.6, repeat: Infinity, delay: i * 0.15 }} />
                        ))}
                      </div>
                    </div>
                  </motion.div>
                )}
              </div>
              <div className="px-6 py-4 border-t border-white/[0.06] flex gap-3">
                <input type="text" value={question} onChange={e => setQuestion(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && sendMessage()}
                  placeholder="Ask anything about your data..."
                  className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-violet-500 transition-colors" />
                <button onClick={sendMessage} disabled={chatLoading || !question.trim()}
                  className="bg-violet-600 hover:bg-violet-500 disabled:opacity-40 text-white px-5 py-2.5 rounded-xl text-sm font-medium transition-all active:scale-95">
                  Send
                </button>
              </div>
            </motion.div>
          )}

        </AnimatePresence>
      </div>
    </div>
  );
}
"use client";
import { useState, useEffect } from "react";
import axios from "axios";
const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
export default function Home() {
  const [datasets, setDatasets] = useState<any[]>([]);
  const [message, setMessage] = useState("");

  useEffect(() => {
    loadDatasets();
  }, []);

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await axios.post(API + "/api/datasets/upload", formData);
      setMessage("Uploaded: " + res.data.name);
      loadDatasets();
    } catch (err) {
      setMessage("Upload failed.");
    }
  }

  async function loadDatasets() {
    const res = await axios.get(API + "/api/datasets/");
    setDatasets(res.data);
  }

  return (
    <div className="min-h-screen bg-[#0f0f13] text-white p-8">
      <h1 className="text-3xl font-bold mb-2">InsightFlow AI</h1>
      <p className="text-gray-400 mb-8">Upload a CSV to get started</p>
      <label className="block border-2 border-dashed border-white/20 rounded-xl p-12 text-center cursor-pointer mb-6">
        <span className="text-gray-400">Click to upload CSV</span>
        <input type="file" accept=".csv" className="hidden" onChange={handleUpload} />
      </label>
      {message && <p className="text-emerald-400 mb-4">{message}</p>}
      <button onClick={loadDatasets} className="text-violet-400 mb-4 block">Refresh list</button>
      {datasets.map((d) => (
        <div key={d.id} className="flex justify-between items-center bg-white/5 rounded-xl p-4 mb-3">
          <div>
            <p className="font-medium">{d.name}</p>
            <p className="text-sm text-gray-400">{d.row_count} rows</p>
          </div>
          <a href={"/dashboard/" + d.id} className="bg-violet-600 text-white px-4 py-2 rounded-lg text-sm">
            Open
          </a>
        </div>
      ))}
    </div>
  );
}

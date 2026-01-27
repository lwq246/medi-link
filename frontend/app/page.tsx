"use client";
import { useState } from "react";

export default function Home() {
  const [status, setStatus] = useState("Ready");
  const [extractedData, setExtractedData] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  // Chat State
  const [question, setQuestion] = useState("");
  const [chatHistory, setChatHistory] = useState<
    { role: string; text: string }[]
  >([]);
  const [chatLoading, setChatLoading] = useState(false);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.[0]) return;
    setLoading(true);
    setStatus("Uploading & Analyzing...");

    const formData = new FormData();
    formData.append("file", e.target.files[0]);

    try {
      const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setExtractedData(data);
      setStatus("Analysis Complete ✅");
    } catch (err) {
      console.error(err);
      setStatus("Error ❌");
    } finally {
      setLoading(false);
    }
  };

  const handleChat = async () => {
    if (!question) return;

    // Add user message to history
    const newHistory = [...chatHistory, { role: "user", text: question }];
    setChatHistory(newHistory);
    setChatLoading(true);
    setQuestion(""); // Clear input

    try {
      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question }),
      });
      const data = await res.json();

      // Add bot response
      setChatHistory([...newHistory, { role: "bot", text: data.answer }]);
    } catch (err) {
      setChatHistory([
        ...newHistory,
        { role: "bot", text: "Error connecting to AI." },
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen flex-col items-center p-12 bg-slate-50 font-sans">
      <h1 className="text-4xl font-bold text-blue-900 mb-8">
        Medi-Link AI Platform
      </h1>

      {/* 1. UPLOAD SECTION */}
      <div className="w-full max-w-2xl bg-white p-6 rounded-xl shadow-md mb-8">
        <label className="font-bold text-gray-700">
          1. Upload Medical Report
        </label>
        <input
          type="file"
          accept=".pdf,.jpg,.png"
          onChange={handleFileUpload}
          className="block w-full mt-2 text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
        />
        {loading && (
          <p className="mt-2 text-blue-600 animate-pulse">{status}</p>
        )}
      </div>

      {/* 2. RESULTS SECTION */}
      {extractedData && (
        <div className="w-full max-w-2xl bg-white p-6 rounded-xl shadow-md mb-8">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">
            📝 AI Analysis
          </h2>
          <div className="p-4 bg-blue-50 rounded-lg border border-blue-100 text-gray-800 whitespace-pre-wrap leading-relaxed">
            {extractedData.ai_analysis}
          </div>
        </div>
      )}

      {/* 3. CHAT SECTION (PHASE 5) */}
      {extractedData && (
        <div className="w-full max-w-2xl bg-white p-6 rounded-xl shadow-md">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">
            💬 Ask a Follow-up
          </h2>

          <div className="mb-4 h-64 overflow-y-auto border p-4 rounded bg-gray-50">
            {chatHistory.length === 0 && (
              <p className="text-gray-400 text-center italic">
                Ask about your results...
              </p>
            )}
            {chatHistory.map((msg, i) => (
              <div
                key={i}
                className={`mb-2 p-3 rounded-lg max-w-[80%] ${msg.role === "user" ? "bg-blue-600 text-white ml-auto" : "bg-gray-200 text-gray-800"}`}
              >
                <strong>{msg.role === "user" ? "You" : "Medi-Link"}:</strong>{" "}
                {msg.text}
              </div>
            ))}
            {chatLoading && (
              <p className="text-gray-500 text-sm animate-pulse">
                AI is typing...
              </p>
            )}
          </div>

          <div className="flex gap-2">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleChat()}
              placeholder="Ex: Is my Hemoglobin too low?"
              className="flex-1 p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-black"
            />
            <button
              onClick={handleChat}
              className="bg-blue-900 text-white px-6 py-3 rounded-lg font-bold hover:bg-blue-800 transition"
            >
              Send
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

"use client";
import { supabase } from "@/lib/supabase";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
export default function Dashboard() {
  const [reports, setReports] = useState<any[]>([]);
  const [selectedReport, setSelectedReport] = useState<any>(null);
  const [status, setStatus] = useState("Ready");
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  // Chat State
  const [question, setQuestion] = useState("");
  const [chatHistory, setChatHistory] = useState<
    { role: string; text: string }[]
  >([]);
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // 1. Auth & Initial Data Fetching
  useEffect(() => {
    const checkUser = async () => {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      if (!session) router.push("/login");
      else fetchReports();
    };
    checkUser();
  }, [router]);

  const fetchReports = async () => {
    const { data } = await supabase
      .from("reports")
      .select("*")
      .order("created_at", { ascending: false });
    setReports(data || []);
  };

  // 2. Fetch Chat History when a report is selected
  const handleSelectReport = async (report: any) => {
    setSelectedReport(report);
    setChatHistory([]); // Clear current view while loading
    setChatLoading(true);

    const { data, error } = await supabase
      .from("chat_messages")
      .select("role, content")
      .eq("report_id", report.id)
      .order("created_at", { ascending: true });

    if (!error && data) {
      setChatHistory(data.map((m) => ({ role: m.role, text: m.content })));
    }
    setChatLoading(false);
  };

  // Scroll chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  // 3. Upload Logic (Synced with Backend Storage)
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.[0]) return;
    setLoading(true);
    setStatus("Analyzing...");

    const file = e.target.files[0];
    const {
      data: { session },
    } = await supabase.auth.getSession();
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        headers: { Authorization: `Bearer ${session?.access_token}` },
        body: formData,
      });
      const data = await res.json();

      // Update local state
      setReports([data, ...reports]); // Add to top of sidebar
      handleSelectReport(data); // View it immediately
      setStatus("Analysis Complete ✅");
    } catch (err) {
      console.error(err);
      alert("Upload failed.");
    } finally {
      setLoading(false);
    }
  };

  // 4. Chat Logic (Sending report_id to Backend)
  const handleChat = async () => {
    if (!question || !selectedReport) return;

    const {
      data: { session },
    } = await supabase.auth.getSession();

    // Optimistic Update: Show user message immediately
    const userMessage = { role: "user", text: question };
    setChatHistory((prev) => [...prev, userMessage]);
    setChatLoading(true);
    const currentQuestion = question;
    setQuestion("");

    try {
      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${session?.access_token}`,
        },
        body: JSON.stringify({
          question: currentQuestion,
          report_id: selectedReport.id, // CRITICAL: Send ID for persistent saving
        }),
      });
      const data = await res.json();
      setChatHistory((prev) => [...prev, { role: "bot", text: data.answer }]);
    } catch (err) {
      setChatHistory((prev) => [
        ...prev,
        { role: "bot", text: "Error connecting to AI." },
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-slate-50 text-slate-900 font-sans overflow-hidden">
      {/* --- SIDEBAR --- */}
      <aside className="w-72 bg-white border-r border-slate-200 flex flex-col shrink-0">
        <div className="p-6">
          <h1 className="text-xl font-bold text-blue-900 flex items-center gap-2">
            <span className="text-2xl">🏥</span> Medi-Link AI
          </h1>
        </div>

        <div className="px-4 mb-4">
          <label className="flex items-center justify-center gap-2 w-full bg-blue-600 hover:bg-blue-700 text-white py-3 px-4 rounded-xl font-bold cursor-pointer transition shadow-md shadow-blue-100">
            <span>+ New Upload</span>
            <input
              type="file"
              className="hidden"
              onChange={handleFileUpload}
              accept=".pdf,.jpg,.png"
            />
          </label>
        </div>

        <nav className="flex-1 overflow-y-auto px-4 space-y-1">
          <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-2 px-2">
            History
          </p>
          {reports.map((report, idx) => (
            <button
              key={report.id ?? idx}
              onClick={() => handleSelectReport(report)}
              className={`w-full text-left p-3 rounded-xl text-sm transition ${
                selectedReport?.id === report.id
                  ? "bg-blue-50 text-blue-900 font-semibold border-l-4 border-blue-600"
                  : "text-slate-600 hover:bg-slate-100"
              }`}
            >
              <p className="truncate">📄 {report.filename}</p>
              <p className="text-[10px] text-slate-400 mt-1">
                {new Date(report.created_at).toLocaleDateString()}
              </p>
            </button>
          ))}
        </nav>

        <div className="p-4 border-t">
          <button
            onClick={async () => {
              await supabase.auth.signOut();
              router.push("/login");
            }}
            className="w-full text-left p-2 text-sm text-red-500 hover:bg-red-50 rounded-lg transition"
          >
            Sign Out
          </button>
        </div>
      </aside>

      {/* --- MAIN WORKSPACE --- */}
      <main className="flex-1 flex flex-col relative overflow-hidden">
        {loading ? (
          <div className="flex-1 flex flex-col items-center justify-center bg-white/50 backdrop-blur-sm z-10">
            <div className="animate-spin text-4xl mb-4">🧬</div>
            <p className="text-blue-900 font-bold animate-pulse">{status}</p>
          </div>
        ) : selectedReport ? (
          <div className="flex-1 flex flex-col h-full">
            <div className="bg-white border-b p-6 flex justify-between items-center">
              <h2 className="text-2xl font-bold text-slate-800">
                {selectedReport.filename}
              </h2>
            </div>

            <div className="flex-1 flex overflow-hidden">
              {/* Left Column: Analysis Results */}
              <section className="w-1/2 p-8 overflow-y-auto border-r border-slate-100">
                <div className="flex items-center gap-2 mb-4">
                  <span className="text-xl">📝</span>
                  <h3 className="font-bold text-lg text-slate-800">
                    AI Medical Summary
                  </h3>
                </div>
                <div className="prose prose-sm max-w-none">
                  <article className="prose prose-slate max-w-none">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        // THIS MAPPING IS WHAT TURNS THE TEXT INTO A REAL TABLE
                        table: (props: any) => (
                          <div className="overflow-x-auto my-6 rounded-lg border border-slate-200">
                            <table
                              className="w-full text-sm text-left border-collapse"
                              {...props}
                            />
                          </div>
                        ),
                        th: (props: any) => (
                          <th
                            className="px-4 py-3 font-bold border-b bg-slate-50 text-blue-900"
                            {...props}
                          />
                        ),
                        td: (props: any) => (
                          <td
                            className="px-4 py-3 border-b border-slate-100"
                            {...props}
                          />
                        ),
                      }}
                    >
                      {selectedReport.ai_analysis}
                    </ReactMarkdown>
                  </article>
                </div>
              </section>

              {/* Right Column: Chat Interface */}
              <section className="w-1/2 flex flex-col bg-slate-50/50 p-8">
                <div className="flex items-center gap-2 mb-4">
                  <span className="text-xl">💬</span>
                  <h3 className="font-bold text-lg text-slate-800">
                    Follow-up Q&A
                  </h3>
                </div>

                <div className="flex-1 bg-white border border-slate-200 rounded-2xl overflow-hidden flex flex-col shadow-sm">
                  <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50/30">
                    {chatHistory.length === 0 && !chatLoading && (
                      <div className="h-full flex items-center justify-center text-slate-400 italic text-sm text-center px-8">
                        Ask about your results...
                      </div>
                    )}
                    {chatHistory.map((msg, i) => (
                      <div
                        key={i}
                        className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                      >
                        <div
                          className={`max-w-[85%] p-3 rounded-2xl text-sm shadow-sm ${
                            msg.role === "user"
                              ? "bg-blue-600 text-white rounded-tr-none"
                              : "bg-white border border-slate-100 text-slate-700 rounded-tl-none"
                          }`}
                        >
                          {msg.text}
                        </div>
                      </div>
                    ))}
                    {chatLoading && (
                      <div className="text-xs text-blue-600 animate-pulse px-2">
                        AI is thinking...
                      </div>
                    )}
                    <div ref={chatEndRef} />
                  </div>

                  <div className="p-4 border-t bg-white">
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={question}
                        onChange={(e) => setQuestion(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && handleChat()}
                        placeholder="Type your question..."
                        className="flex-1 p-3 bg-slate-100 border-none rounded-xl focus:ring-2 focus:ring-blue-500 text-sm text-black"
                      />
                      <button
                        onClick={handleChat}
                        className="bg-blue-900 text-white px-5 py-3 rounded-xl hover:bg-blue-800 transition"
                      >
                        Send
                      </button>
                    </div>
                  </div>
                </div>
              </section>
            </div>
          </div>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center p-12 text-center">
            <div className="w-24 h-24 bg-blue-50 rounded-full flex items-center justify-center text-4xl mb-6">
              📂
            </div>
            <h2 className="text-2xl font-bold text-slate-800 mb-2">
              No Report Selected
            </h2>
            <p className="text-slate-500 max-w-sm">
              Upload a medical report or select one from history.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}

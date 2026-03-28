"use client";
import { supabase } from "@/lib/supabase";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

export default function LandingPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);

  // SMART REDIRECT: If logged in, go straight to Dashboard
  useEffect(() => {
    const checkUser = async () => {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      if (session) {
        router.push("/dashboard");
      } else {
        setLoading(false);
      }
    };
    checkUser();
  }, [router]);

  if (loading)
    return (
      <div className="h-screen flex items-center justify-center">
        Loading...
      </div>
    );

  return (
    <div className="min-h-screen bg-white font-sans text-slate-900">
      {/* Navigation */}
      <nav className="flex justify-between items-center p-6 max-w-7xl mx-auto">
        <h1 className="text-2xl font-bold text-blue-900">Medi-Link AI</h1>
        <Link
          href="/login"
          className="bg-blue-900 text-white px-6 py-2 rounded-full font-semibold hover:bg-blue-800 transition"
        >
          Sign In
        </Link>
      </nav>

      {/* Hero Section */}
      <header className="flex flex-col items-center text-center mt-20 px-6">
        <h2 className="text-5xl md:text-6xl font-extrabold text-blue-900 mb-6 leading-tight">
          Understand Your Medical <br /> Reports in Seconds.
        </h2>
        <p className="text-xl text-slate-600 max-w-2xl mb-10 leading-relaxed">
          Upload blood tests, MRI results, or any medical document. Our AI
          identifies abnormal values and explains your health in simple
          language.
        </p>
        <Link
          href="/login"
          className="bg-blue-600 text-white px-10 py-4 rounded-xl text-lg font-bold shadow-lg hover:bg-blue-700 transition transform hover:scale-105"
        >
          Get Started for Free
        </Link>
      </header>

      {/* Feature Section */}
      <section className="mt-32 grid md:grid-cols-3 gap-8 max-w-6xl mx-auto px-6 pb-20">
        <div className="p-8 bg-blue-50 rounded-2xl">
          <span className="text-4xl">📄</span>
          <h3 className="text-xl font-bold mt-4 mb-2">AI Summary</h3>
          <p className="text-slate-600">
            Complex medical jargon translated into plain English you can
            actually understand.
          </p>
        </div>
        <div className="p-8 bg-blue-50 rounded-2xl">
          <span className="text-4xl">🚩</span>
          <h3 className="text-xl font-bold mt-4 mb-2">Flag Abnormalities</h3>
          <p className="text-slate-600">
            Instantly see which markers are outside of standard reference
            ranges.
          </p>
        </div>
        <div className="p-8 bg-blue-50 rounded-2xl">
          <span className="text-4xl">💬</span>
          <h3 className="text-xl font-bold mt-4 mb-2">Follow-up Chat</h3>
          <p className="text-slate-600">
            Ask the AI questions about your specific results anytime, 24/7.
          </p>
        </div>
      </section>

      <footer className="text-center py-10 text-slate-400 text-sm">
        © 2024 Medi-Link AI. Not a replacement for professional medical advice.
      </footer>
    </div>
  );
}

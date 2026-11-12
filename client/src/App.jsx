import { useEffect, useState } from "react";
import LandingPage from "./pages/LandingPage";
import AboutPage from "./pages/AboutPage";
import HowItWorksPage from "./pages/HowItWorksPage";
import PredictionPage from "./pages/PredictionPage";

export default function App() {
  const [pathname, setPathname] = useState(() => {
    if (typeof window !== "undefined") {
      return window.location.pathname;
    }
    return "/";
  });

  useEffect(() => {
    const handlePopState = () => {
      setPathname(window.location.pathname);
    };
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  const path = pathname.replace(/\/$/, "") || "/";

  if (path === "/about") {
    return <AboutPage />;
  }

  if (path === "/how-it-works") {
    return <HowItWorksPage />;
  }

  if (path === "/predict" || path === "/demo") {
    return <PredictionPage />;
  }

  return <LandingPage />;
}

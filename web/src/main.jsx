import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import "./styles.css";

document.documentElement.dataset.app = "pdf-rag";
ReactDOM.createRoot(document.getElementById("root")).render(<App />);

import { useState } from "react";
import ModeSelector from "./components/ModeSelector";
import ConversationView from "./components/ConversationView";
import AnimatedBackground from "./components/AnimatedBackground";

function App() {
  const [mode, setMode] = useState(null);

  return (
    <div className="relative min-h-screen">
      <AnimatedBackground />
      {!mode ? (
        <ModeSelector onSelectMode={setMode} />
      ) : (
        <ConversationView mode={mode} onBack={() => setMode(null)} />
      )}
    </div>
  );
}

export default App;
// Top-level imports and new panes
import React, { useState, useEffect } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";

import GridWorldCanvas from "./components/GridWorldCanvas.jsx";
import useGridWorld from "./hooks/useGridWorld.js";
import useDQN from "./hooks/useDQN.js";
import CartPoleCanvas from "./components/CartPoleCanvas.jsx";
import useCartPoleDQN from "./hooks/useCartPoleDQN.js";

// GridWorld view pane extracted from old App
function GridWorldPane() {
  const { state, reset, step } = useGridWorld({ gridSize: 6, cellSize: 72 });
  const {
    startTraining,
    stopTraining,
    training,
    episode,
    totalSteps,
    lastReward,
    epsilon,
    avgReward,
  } = useDQN({ gridState: state, envStep: step, envReset: reset });

  useEffect(() => {
    reset();
  }, [reset]);

  return (
    <div className="flex flex-col items-center">
      <h2 className="text-2xl font-semibold mt-4">GridWorld DQN</h2>
      <div className="mt-6">
        <GridWorldCanvas
          gridSize={state.gridSize}
          cellSize={state.cellSize}
          agentPos={state.agentPos}
          blocks={state.blocks}
          pit={state.pit}
          goal={state.goalPos}
        />
      </div>
      <div className="mt-6 flex gap-3">
        <button
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded"
          onClick={reset}
          disabled={training}
        >
          Reset
        </button>
        <button
          className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded"
          onClick={() => startTraining({ episodes: 100 })}
          disabled={training}
        >
          Start Training
        </button>
        <button
          className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded"
          onClick={stopTraining}
          disabled={!training}
        >
          Stop
        </button>
      </div>
      <div className="mt-6 text-gray-300">
        <div>Episode: {episode}</div>
        <div>Total Steps: {totalSteps}</div>
        <div>Last Episode Reward: {lastReward.toFixed(3)}</div>
        <div>Epsilon: {epsilon.toFixed(3)}</div>
        <div>Avg Reward (10): {avgReward.toFixed(3)}</div>
      </div>
    </div>
  );
}

// New CartPole view pane
function CartPolePane() {
  const {
    startTraining,
    stopTraining,
    training,
    episode,
    totalSteps,
    lastReward,
    epsilon,
    avgReward,
    obs,
    resetEnv,
  } = useCartPoleDQN();

  useEffect(() => {
    resetEnv();
  }, [resetEnv]);

  return (
    <div className="flex flex-col items-center">
      <h2 className="text-2xl font-semibold mt-4">CartPole DQN</h2>
      <div className="mt-6">
        <CartPoleCanvas obs={obs} />
      </div>
      <div className="mt-6 flex gap-3">
        <button
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded"
          onClick={resetEnv}
          disabled={training}
        >
          Reset
        </button>
        <button
          className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded"
          onClick={() => startTraining({ episodes: 1000 })}
          disabled={training}
        >
          Start Training
        </button>
        <button
          className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded"
          onClick={stopTraining}
          disabled={!training}
        >
          Stop
        </button>
      </div>
      <div className="mt-6 text-gray-300">
        <div>Episode: {episode}</div>
        <div>Total Steps: {totalSteps}</div>
        <div>Last Episode Reward: {lastReward.toFixed(3)}</div>
        <div>Epsilon: {epsilon.toFixed(3)}</div>
        <div>Avg Reward (10): {avgReward.toFixed(3)}</div>
      </div>
    </div>
  );
}

function App() {
  const [mode, setMode] = useState("grid");

  return (
    <div className="h-screen w-screen bg-gray-900 text-white flex flex-col items-center">
      <h1 className="text-3xl font-bold mt-6">DQN Examples (js-pytorch)</h1>

      <div className="mt-4">
        <label className="mr-3 font-medium">Example:</label>
        <select
          className="px-3 py-2 bg-gray-800 border border-gray-700 rounded"
          value={mode}
          onChange={(e) => setMode(e.target.value)}
        >
          <option value="grid">GridWorld</option>
          <option value="cartpole">CartPole (Balance Broom)</option>
        </select>
      </div>

      {mode === "grid" ? <GridWorldPane /> : <CartPolePane />}
    </div>
  );
}

const root = createRoot(document.getElementById("root"));
root.render(<App />);

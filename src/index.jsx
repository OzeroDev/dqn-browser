// Top-level imports and new panes
import React, { useState, useEffect, useRef } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";

import GridWorldCanvas from "./components/GridWorldCanvas.jsx";
import useGridWorld from "./hooks/useGridWorld.js";
import useDQN, { useNewDQN } from "./hooks/useDQN.js";
import useTableQLearning from "./hooks/useTableQLearning.js";
import CartPoleCanvas from "./components/CartPoleCanvas.jsx";
import useCartPoleDQN from "./hooks/useCartPoleDQN.js";
import NNVisual from "./components/NNVisual.jsx";
import DQNExplain from "./components/DQNExplain.jsx";

// Tooltip component for hyperparameters
function HyperparamLabel({ children, tooltip, targetId }) {
  const [showTooltip, setShowTooltip] = useState(false);
  
  const handleClick = () => {
    const element = document.getElementById(targetId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      // Add a brief highlight effect
      element.classList.add('highlight-flash');
      setTimeout(() => element.classList.remove('highlight-flash'), 2000);
    }
  };
  
  return (
    <span className="relative inline-block">
      <span
        className="cursor-pointer hover:text-blue-400 transition-colors underline decoration-dotted"
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        onClick={handleClick}
      >
        {children}
      </span>
      {showTooltip && (
        <span className="absolute z-50 bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-slate-950 text-slate-200 text-xs rounded border border-slate-700 shadow-lg whitespace-nowrap">
          {tooltip}
          <span className="absolute top-full left-1/2 transform -translate-x-1/2 -mt-1 border-4 border-transparent border-t-slate-950"></span>
        </span>
      )}
    </span>
  );
}

// Interactive view pane
function InteractivePane() {
  const { state, reset, step } = useGridWorld({ gridSize: 6, cellSize: 72 });
  // Hidden layer config state
  const [numHidden, setNumHidden] = useState(2);
  const [hiddenSizes, setHiddenSizes] = useState([64, 64]);
  // Hyperparameter config state
  const [learningRate, setLearningRate] = useState(5e-4);
  const [epsilonDecay, setEpsilonDecay] = useState(1000);
  const [gamma, setGamma] = useState(0.99);

  // Update hiddenSizes when numHidden changes
  useEffect(() => {
    setHiddenSizes((prev) => {
      if (numHidden > prev.length) {
        return [...prev, ...Array(numHidden - prev.length).fill(64)];
      } else if (numHidden < prev.length) {
        return prev.slice(0, numHidden);
      }
      return prev;
    });
  }, [numHidden]);

  const {
    startTraining,
    stopTraining,
    training,
    episode,
    totalSteps,
    lastReward,
    epsilon,
    avgReward,
    getQValues,
    exploreCount,
    exploitCount,
    model,
  } = useNewDQN({ 
    gridState: state, 
    envStep: step, 
    envReset: reset, 
    hiddenLayers: hiddenSizes,
    learningRate,
    epsilonDecay,
    gamma
  });

  const [cellQGrid, setCellQGrid] = useState(null);

  // compute Q-values for every valid cell whenever grid state or model changes
  useEffect(() => {
    if (!state || !getQValues) {
      setCellQGrid(null);
      return;
    }

    const gs = state.gridSize;
    const grid = Array.from({ length: gs }, () =>
      Array.from({ length: gs }, () => null)
    );

    for (let r = 0; r < gs; r++) {
      for (let c = 0; c < gs; c++) {
        const isBlock = state.blocks.some((b) => b[0] === r && b[1] === c);
        const isPit = state.pit[0] === r && state.pit[1] === c;
        const isGoal = state.goalPos[0] === r && state.goalPos[1] === c;
        if (isBlock || isPit || isGoal) {
          grid[r][c] = null;
          continue;
        }
        try {
          const q = getQValues([r, c]);
          grid[r][c] = Array.isArray(q) ? q : null;
        } catch (e) {
          grid[r][c] = null;
        }
      }
    }

    setCellQGrid(grid);
  }, [state, getQValues]);

  useEffect(() => {
    reset();
  }, [reset]);

  return (
    <div className="flex flex-col overflow-hidden w-full h-full items-center p-8">
      {/* Top Control Panel */}
      <div className="flex flex-col items-center mb-8">
        <div className="flex gap-3 mb-4">
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
        {/* Hidden layer controls */}
        <div className="flex gap-4 items-center mb-4 flex-wrap">
          <label className="text-gray-300 text-sm">Number of Layers:</label>
          <select
            value={numHidden}
            onChange={e => setNumHidden(Number(e.target.value))}
            disabled={training}
            className="bg-slate-800 text-white px-2 py-1 rounded"
          >
            {[1, 2, 3].map(n => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
          {hiddenSizes.map((size, idx) => {
            const sizeOptions = [8, 16, 32, 64, 128];
            return (
              <div key={idx} className="flex items-center gap-1">
                <label className="text-gray-300 text-xs">Layer {idx+1}:</label>
                <select
                  value={size}
                  disabled={training}
                  onChange={e => {
                    setHiddenSizes(sizes => sizes.map((s, i) => i === idx ? Number(e.target.value) : s));
                  }}
                  className="bg-slate-800 text-white px-2 py-1 rounded border border-slate-700"
                >
                  {sizeOptions.map(opt => (
                    <option key={opt} value={opt}>{opt}</option>
                  ))}
                </select>
              </div>
            );
          })}
        </div>
        {/* Hyperparameter controls */}
        <div className="flex gap-4 items-center mb-4 flex-wrap">
          <label className="text-gray-300 text-sm">
            <HyperparamLabel tooltip="Step size for network weight updates" targetId="learning-rate-section">
              Learning Rate
            </HyperparamLabel>:
          </label>
          <select
            value={learningRate}
            onChange={e => setLearningRate(Number(e.target.value))}
            disabled={training}
            className="bg-slate-800 text-white px-2 py-1 rounded"
          >
            {[0.00001, 0.00005, 0.0001, 0.0005, 0.001].map(val => (
              <option key={val} value={val}>{val}</option>
            ))}
          </select>
          <label className="text-gray-300 text-sm">
            <HyperparamLabel tooltip="How quickly exploration decreases" targetId="epsilon-decay-section">
              Epsilon Decay
            </HyperparamLabel>:
          </label>
          <select
            value={epsilonDecay}
            onChange={e => setEpsilonDecay(Number(e.target.value))}
            disabled={training}
            className="bg-slate-800 text-white px-2 py-1 rounded"
          >
            {[500, 1000, 2000, 5000, 10000].map(val => (
              <option key={val} value={val}>{val}</option>
            ))}
          </select>
          <label className="text-gray-300 text-sm">
            <HyperparamLabel tooltip="Discount factor for future rewards" targetId="gamma-section">
              Gamma
            </HyperparamLabel>:
          </label>
          <select
            value={gamma}
            onChange={e => setGamma(Number(e.target.value))}
            disabled={training}
            className="bg-slate-800 text-white px-2 py-1 rounded"
          >
            {[0.90, 0.95, 0.99, 0.999].map(val => (
              <option key={val} value={val}>{val}</option>
            ))}
          </select>
        </div>
        <div className="flex gap-6 text-gray-300 text-sm">
          <div>Episode: {episode}</div>
          <div>Total Steps: {totalSteps}</div>
          <div>Last Episode Reward: {lastReward.toFixed(3)}</div>
          <div>Epsilon: {epsilon.toFixed(3)}</div>
          <div>Avg Reward (10): {avgReward.toFixed(3)}</div>
        </div>
      </div>

      {/* Three Grid Layout */}
      <div className="flex justify-center gap-8 mb-8">
        {/* Left Panel - Q-value arrows only */}
        <div className="flex flex-col items-center">
          <h3 className="text-xl font-semibold mb-4">Q-Value Arrows</h3>
          <GridWorldCanvas
            gridSize={state.gridSize}
            cellSize={state.cellSize}
            agentPos={state.agentPos}
            blocks={state.blocks}
            pit={state.pit}
            goal={state.goalPos}
            actionQGrid={cellQGrid}
            directionLidarFlag={false}
          />
        </div>

        {/* Center Panel - Base GridWorld (no arrows/lines) */}
        <div className="flex flex-col items-center">
          <h3 className="text-xl font-semibold mb-4">Base Environment</h3>
          <GridWorldCanvas
            gridSize={state.gridSize}
            cellSize={state.cellSize}
            agentPos={state.agentPos}
            blocks={state.blocks}
            pit={state.pit}
            goal={state.goalPos}
            actionQGrid={null}
            directionLidarFlag={false}
          />
        </div>

        {/* Right Panel - Agent visualizations (lidar, goal line, pit circle) */}
        <div className="flex flex-col items-center">
          <h3 className="text-xl font-semibold mb-4">Agent Sensors</h3>
          <GridWorldCanvas
            gridSize={state.gridSize}
            cellSize={state.cellSize}
            agentPos={state.agentPos}
            blocks={state.blocks}
            pit={state.pit}
            goal={state.goalPos}
            actionQGrid={null}
            directionLidarFlag={true}
          />
        </div>
      </div>

      {/* Neural Network Panel - Below the grids */}
      <div className="flex flex-col items-center w-full">
        <h3 className="text-xl font-semibold mb-4">Neural Network</h3>
        <div className="w-full flex justify-center">
          <NNVisual model={model} refreshKey={totalSteps} width={1400} height={600} />
        </div>
      </div>
    </div>
  );
}


function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <InteractivePane />
      <DQNExplain />
    </div>
  );
}

const root = createRoot(document.getElementById("root"));
root.render(<App />);

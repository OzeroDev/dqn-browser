// Top-level imports and new panes
import React, { useState, useEffect } from "react";
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

// Interactive view pane
function InteractivePane() {
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
    getQValues,
    exploreCount,
    exploitCount,
    model,
  } = useNewDQN({ gridState: state, envStep: step, envReset: reset });

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

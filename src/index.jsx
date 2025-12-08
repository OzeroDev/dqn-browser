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
    <div className="flex overflow-hidden w-full h-full justify-center ">
      <div className="overflow-y-hidden h-screen">
        <h2 className="text-2xl font-semibold mt-6">GridWorld Deep Q</h2>
        <div className="mt-6">
          <GridWorldCanvas
            gridSize={state.gridSize}
            cellSize={state.cellSize}
            agentPos={state.agentPos}
            blocks={state.blocks}
            pit={state.pit}
            goal={state.goalPos}
            actionQGrid={cellQGrid}
            directionLidarFlag={true}
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
      <div className="flex-1 overflow-y-auto h-screen">
        <NNVisual model={model} refreshKey={totalSteps} />
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

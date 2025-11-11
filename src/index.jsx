// Top-level imports and new panes
import React, { useState, useEffect } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";

import GridWorldCanvas from "./components/GridWorldCanvas.jsx";
import useGridWorld from "./hooks/useGridWorld.js";
import useDQN from "./hooks/useDQN.js";
import useTableQLearning from "./hooks/useTableQLearning.js";
import CartPoleCanvas from "./components/CartPoleCanvas.jsx";
import useCartPoleDQN from "./hooks/useCartPoleDQN.js";
// GridWorld view pane for Deep Q (DQN)
function GridWorldDQNPane() {
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
  } = useDQN({ gridState: state, envStep: step, envReset: reset });

  const [cellQGrid, setCellQGrid] = useState(null);

  // compute Q-values for every valid cell whenever grid state or model changes
  useEffect(() => {
    if (!state || !getQValues) {
      setCellQGrid(null);
      return;
    }

    const gs = state.gridSize;
    const grid = Array.from({ length: gs }, () => Array.from({ length: gs }, () => null));

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
    <div className="flex flex-col items-center">
      <h2 className="text-2xl font-semibold mt-4">GridWorld Deep Q</h2>
      <div className="mt-6">
        <GridWorldCanvas
          gridSize={state.gridSize}
          cellSize={state.cellSize}
          agentPos={state.agentPos}
          blocks={state.blocks}
          pit={state.pit}
          goal={state.goalPos}
          actionQGrid={cellQGrid}
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

// GridWorld view pane for Table Q-learning
function GridWorldTablePane() {
  const { state, reset, step } = useGridWorld({ gridSize: 6, cellSize: 72 });
  const {
    startTraining: startTableTraining,
    stopTraining: stopTableTraining,
    training: tableTraining,
    episode: tableEpisode,
    totalSteps: tableTotalSteps,
    lastReward: tableLastReward,
    epsilon: tableEpsilon,
    avgReward: tableAvgReward,
    getQValues: getTableQValues,
    exploreCount: tableExploreCount,
    exploitCount: tableExploitCount,
  } = useTableQLearning({ gridState: state, envStep: step, envReset: reset });

  const [cellTableQGrid, setCellTableQGrid] = useState(null);

  useEffect(() => {
    if (!state || !getTableQValues) {
      setCellTableQGrid(null);
      return;
    }

    const gs = state.gridSize;
    const grid = Array.from({ length: gs }, () => Array.from({ length: gs }, () => null));

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
          const q = getTableQValues([r, c]);
          grid[r][c] = Array.isArray(q) ? q : null;
        } catch (e) {
          grid[r][c] = null;
        }
      }
    }

    setCellTableQGrid(grid);
  }, [state, getTableQValues]);

  useEffect(() => {
    reset();
  }, [reset]);

  return (
    <div className="flex flex-col items-center">
      <h2 className="text-2xl font-semibold mt-4">GridWorld Table Q</h2>
      <div className="mt-6">
        <GridWorldCanvas
          gridSize={state.gridSize}
          cellSize={state.cellSize}
          agentPos={state.agentPos}
          blocks={state.blocks}
          pit={state.pit}
          goal={state.goalPos}
          actionQGrid={cellTableQGrid}
        />
      </div>
      <div className="mt-6 flex gap-3">
        <button
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded"
          onClick={reset}
          disabled={tableTraining}
        >
          Reset
        </button>
        <button
          className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded"
          onClick={() => startTableTraining({ episodes: 500 })}
          disabled={tableTraining}
        >
          Start Training
        </button>
        <button
          className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded"
          onClick={stopTableTraining}
          disabled={!tableTraining}
        >
          Stop
        </button>
      </div>
      <div className="mt-6 text-gray-300">
        <div>Episode: {tableEpisode}</div>
        <div>Total Steps: {tableTotalSteps}</div>
        <div>Last Episode Reward: {tableLastReward.toFixed(3)}</div>
        <div>Epsilon: {tableEpsilon.toFixed(3)}</div>
        <div>Avg Reward (10): {tableAvgReward.toFixed(3)}</div>
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
  const [mode, setMode] = useState("grid-dqn");

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
          <option value="grid-dqn">GridWorld (Deep Q-Learning)</option>
          <option value="grid-table">GridWorld (Q-Learning)</option>
          <option value="cartpole">CartPole</option>
        </select>
      </div>

      {mode === "grid-dqn" && <GridWorldDQNPane />}
      {mode === "grid-table" && <GridWorldTablePane />}
      {mode === "cartpole" && <CartPolePane />}
    </div>
  );
}

const root = createRoot(document.getElementById("root"));
root.render(<App />);

# DQN Examples (js-pytorch): GridWorld & CartPole

A React + js-pytorch web app demonstrating Deep Q-Learning on two environments:
- GridWorld: simple navigation with obstacles, pit, and goal.
- CartPole: balance a pole on a moving cart via continuous physics.

## Features
- Real-time canvas rendering for both environments.
- DQN with replay memory and epsilon-greedy exploration.
- GPU-backed tensors via `js-pytorch` (falls back where appropriate).
- Simple UI to reset, start, and stop training; live stats display.

## Requirements
- Node.js 18+ and npm.
- Windows environment (commands below use Windows syntax).
- Modern browser (Chrome/Edge recommended).

## Install
```bash
npm install
```

## Start Dev Server
```bash
npm run start
```
- Opens `http://localhost:3000/`.
- Uses `webpack-dev-server` with CSP headers configured in `webpack.config.js`.
- The app is served via `src/index.html` which loads `/bundle.js`.

## Build
```bash
npm run build
```
- Outputs production bundle to `dist/`.

## Project Structure
- `src/components/` — Canvas renderers:
  - `GridWorldCanvas.jsx`
  - `CartPoleCanvas.jsx`
- `src/hooks/` — RL and environment logic:
  - `useDQN.js` — GridWorld-specific DQN with feature engineering.
  - `useCartPoleDQN.js` — CartPole DQN and physics sim.
  - `useGridWorld.js` — GridWorld environment and React hook wrapper.
- `src/index.jsx` — App root, panes, and UI.
- `webpack.config.js` — Dev server, loaders, CSP headers.
- `tailwind.config.js`, `postcss.config.js`, `index.css` — Styles.

## Usage
- Use the dropdown to switch between “GridWorld” and “CartPole”.
- Click “Reset” to reset environment state.
- Click “Start Training” to run DQN for the specified episodes.
- Click “Stop” to halt training.
- Stats shown: episode, total steps, last episode reward, epsilon, average reward (window=10).

## Notes
- The DQN networks are defined in `useDQN.js` and `useCartPoleDQN.js`.
- Replay memory size: 10,000; batch size: 64.
- Epsilon decays are tuned per environment (`EPS_DECAY`).
- GridWorld features include normalized position and distances to goal/pit.

## License
This project is provided as-is for learning purposes. Add a license if you plan to distribute.
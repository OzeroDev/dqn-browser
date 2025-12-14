# Deep Q-Network Visualization: GridWorld

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
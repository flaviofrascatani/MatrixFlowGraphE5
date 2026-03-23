# Matrix Flow Graph — E5 Transformer

Interactive visualization of matrix multiplications through a transformer architecture (E5-multilingual-base style).

Watch real matrices get projected into Q, K, V in parallel, merged into attention scores, and transformed through FFN layers — with exact dimensions shown at every step.

![Preview](https://img.shields.io/badge/Live-Demo-d4a840?style=for-the-badge)

## 🚀 Deploy on Vercel (easiest)

1. Push all these files to your GitHub repo
2. Go to [vercel.com](https://vercel.com) → **Add New Project**
3. Import your `MatrixFlowGraphE5` repo
4. Vercel auto-detects Vite — just click **Deploy**
5. Done! You get a link like `matrix-flow-graph-e5.vercel.app`

## 💻 Run locally

```bash
npm install
npm run dev
```

Opens at `http://localhost:5173`

## 📁 Project structure

```
├── index.html          # Entry HTML
├── package.json        # Dependencies (React + Vite)
├── vite.config.js      # Vite configuration
└── src/
    ├── main.jsx        # React mount point
    └── App.jsx         # Main application (all logic here)
```

## Features

- **Parallel Q/K/V flow** — shows the Y-shaped data flow, not a misleading linear pipeline
- **Real matrix math** — actual multiplications with visible values
- **Configurable** — adjust seq length, d_model, heads, layers, FF multiplier
- **Click to inspect** — full matrix view with statistics (min, max, mean, std)
- **Dimension journey** — see exactly where matrices expand and contract

## License

MIT

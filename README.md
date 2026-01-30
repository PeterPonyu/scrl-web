# scRL-Web

A web interface for **scRL** (Single-cell Reinforcement Learning) - an advanced computational framework that leverages reinforcement learning to decode cellular fate decisions from single-cell RNA sequencing data.

**🌐 Live Demo**: [https://scrl-web.vercel.app](https://scrl-web.vercel.app)

## Features

- **Interactive Web UI**: User-friendly interface for single-cell analysis
- **Complete Pipeline**: Data upload → Preprocessing → Grid Embedding → RL Training → Visualization
- **Real-time Visualization**: Cell embeddings color-coded by cluster, pseudotime, or fate values
- **Multiple Analysis Modes**: Supports both lineage-based (discrete) and gene-based (continuous) rewards
- **Decision & Contribution Modes**: Identify early decision points and late contributions to cell fate
- **Bilingual Support**: English and Chinese interface

## Quick Start

### Online Usage

Visit [https://scrl-web.vercel.app](https://scrl-web.vercel.app) to use the application directly.

### Local Development

#### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

#### Frontend

```bash
cd frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

Visit http://localhost:3000 to access the application.

## Architecture

```
scrl-web/
├── frontend/          # Next.js 14 + React + Tailwind CSS
│   ├── src/
│   │   ├── app/       # Next.js App Router pages
│   │   ├── components/ # React components
│   │   └── lib/       # API client and utilities
│   └── package.json
│
├── backend/           # FastAPI + Python
│   ├── app/
│   │   ├── main.py    # FastAPI application
│   │   ├── routers/   # API endpoints
│   │   ├── services/  # scRL integration
│   │   └── models/    # Pydantic schemas
│   └── requirements.txt
│
└── README.md
```

## Usage Workflow

### 1. Upload Data
- Upload your `.h5ad` (AnnData) or `.csv` file
- Or use the demo dataset (Paul15 mouse hematopoiesis)

### 2. Preprocess
- Configure embedding method (t-SNE or UMAP)
- Set PCA components and clustering resolution
- Run preprocessing pipeline

### 3. Generate Grid
- Transform embedding into a navigable grid
- Adjust grid resolution for different levels of detail

### 4. Align Pseudotime
- Select the starting cluster (stem/progenitor cells)
- Pseudotime calculated using shortest paths on the grid

### 5. Configure Rewards
- **Discrete/Lineage-based**: Select start and end clusters
- **Continuous/Gene-based**: Select genes for reward calculation
- Choose between Decision (early) and Contribution (late) modes

### 6. Train Model
- Actor-Critic algorithm learns optimal policies
- Training progress displayed in real-time
- Results show fate decision strength at each state

## Deployment

### Backend (Render)

- **URL**: https://scrl-api.onrender.com
- **Root Directory**: `backend`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### Frontend (Vercel)

- **URL**: https://scrl-web.vercel.app
- **Root Directory**: `frontend`
- **Environment Variable**: `NEXT_PUBLIC_API_URL=https://scrl-api.onrender.com`

## API Documentation

Visit https://scrl-api.onrender.com/docs for the complete API documentation.

### Main Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /api/data/session` | Create analysis session |
| `POST /api/data/upload/{session_id}` | Upload data file |
| `POST /api/data/demo/{dataset}` | Load demo dataset |
| `POST /api/analysis/preprocess/{session_id}` | Data preprocessing |
| `POST /api/analysis/grid/{session_id}` | Generate grid embedding |
| `POST /api/analysis/pseudotime/{session_id}` | Calculate pseudotime |
| `POST /api/training/configure/{session_id}` | Configure rewards |
| `POST /api/training/train/{session_id}` | Train RL model |
| `GET /api/results/plot/{session_id}` | Get visualization data |

## Scientific Background

scRL treats cell development like a strategic decision-making game. The RL agent navigates through the cellular state space, learning which states represent strong fate commitments.

Key findings from scRL:
- Identifies early decision points before cells show obvious lineage commitment
- Outperforms 15+ state-of-the-art trajectory inference methods
- Reveals previously unknown regulatory factors controlling fate decisions

## Citation

If you use scRL in your research, please cite:

```bibtex
@article{fu2025scrl,
  title={scRL: Utilizing Reinforcement Learning to Evaluate Fate Decisions in Single-Cell Data},
  author={Fu, Zeyu and Chen, Chunlin and Wang, Song and Wang, Junping and Chen, Shilei},
  journal={Biology},
  volume={14},
  number={6},
  pages={679},
  year={2025},
  doi={10.3390/biology14060679}
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## Contact

- **Author**: Zeyu Fu
- **Email**: fuzeyu99@126.com
- **Original scRL**: [GitHub](https://github.com/PeterPonyu/scRL)

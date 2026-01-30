# scRL-Web

A web interface for **scRL** (Single-cell Reinforcement Learning) - an advanced computational framework that leverages reinforcement learning to decode cellular fate decisions from single-cell RNA sequencing data.

## 🌟 Features

- **Interactive Web UI**: User-friendly interface for single-cell analysis
- **Complete Pipeline**: Data upload → Preprocessing → Grid Embedding → RL Training → Visualization
- **Real-time Visualization**: See your cell embeddings color-coded by cluster, pseudotime, or fate values
- **Multiple Analysis Modes**: Supports both lineage-based (discrete) and gene-based (continuous) rewards
- **Decision & Contribution Modes**: Identify both early decision points and late contributions to cell fate

## 🏗️ Architecture

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

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- npm or yarn

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to access the application.

## 📖 Usage

### 1. Upload Data
- Upload your `.h5ad` (AnnData) or `.csv` file
- Or use the demo dataset (Paul15 mouse hematopoiesis)

### 2. Preprocess
- Configure embedding method (t-SNE or UMAP)
- Set PCA components and clustering resolution
- Run preprocessing pipeline

### 3. Generate Grid
- The algorithm transforms your embedding into a navigable grid
- Adjust grid resolution for different levels of detail

### 4. Align Pseudotime
- Select the starting cluster (stem/progenitor cells)
- Pseudotime is calculated using shortest paths on the grid

### 5. Configure Rewards
- **Discrete/Lineage-based**: Select start and end clusters
- **Continuous/Gene-based**: Select genes for reward calculation
- Choose between Decision (early) and Contribution (late) modes

### 6. Train Model
- Actor-Critic algorithm learns optimal policies
- Training progress is displayed in real-time
- Results show fate decision strength at each state

## 🌐 Deployment

### Deploy Backend to Render

1. Create a new Web Service on [Render](https://render.com)
2. Connect your GitHub repository
3. Configure:
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### Deploy Frontend to Vercel

1. Import project to [Vercel](https://vercel.com)
2. Configure:
   - **Root Directory**: `frontend`
   - **Environment Variables**:
     - `NEXT_PUBLIC_API_URL`: Your Render backend URL

## 🔬 Scientific Background

scRL treats cell development like a strategic decision-making game. The RL agent navigates through the cellular state space, learning which states represent strong fate commitments.

Key findings from scRL:
- Identifies early decision points before cells show obvious lineage commitment
- Outperforms 15+ state-of-the-art trajectory inference methods
- Reveals previously unknown regulatory factors controlling fate decisions

## 📚 Citation

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

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## 📧 Contact

- **Author**: Zeyu Fu
- **Email**: fuzeyu99@126.com
- **Original scRL**: [GitHub](https://github.com/PeterPonyu/scRL)

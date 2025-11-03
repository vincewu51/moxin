# Deployment Guide: Qdrant Cloud + HuggingFace Spaces

This guide walks you through deploying Moxin's novel query system online using:
- **Qdrant Cloud** for vector database hosting (free 1GB tier)
- **HuggingFace Spaces** for web interface hosting (free)

## Prerequisites

✅ You have:
- OpenRouter embeddings indexed locally (`data/embeddings_openrouter/`)
- Qdrant API key (already in `.env`)
- HuggingFace API token (already in `.env`)

## Step 1: Set Up Qdrant Cloud Cluster

### 1.1 Create a Free Cluster

1. Go to [Qdrant Cloud Console](https://cloud.qdrant.io/)
2. Sign in or create an account
3. Click **"Create Cluster"**
4. Select **"Free Tier"** (1GB storage, perfect for your 12,410 chunks)
5. Choose a cluster name (e.g., `moxin-novel`)
6. Select a region (choose closest to you for best performance)
7. Click **"Create"**

### 1.2 Get Your Cluster URL

After cluster creation:
1. Go to your cluster dashboard
2. Copy the **Cluster URL** (looks like: `https://abc123.us-east-1-0.aws.cloud.qdrant.io:6333`)
3. Update your `.env` file:

```bash
QDRANT_URL=https://your-cluster-url.qdrant.io:6333
```

### 1.3 Verify API Key

Your Qdrant API key should already be in `.env`:
```bash
QDRANT_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.PXAHs3rw3AR8yQHIMvR9Hvp5KZll179ao3BEDbtXt64
```

## Step 2: Migrate Embeddings to Qdrant Cloud

### 2.1 Install Qdrant Client

```bash
source .venv/bin/activate
uv pip install qdrant-client
```

### 2.2 Run Migration Script

```bash
# Migrate from ChromaDB to Qdrant Cloud
python src/deploy/migrate_to_qdrant.py \
  --chromadb data/embeddings_openrouter \
  --collection novel_chunks_openrouter \
  --qdrant-collection novel_chunks_openrouter
```

This will:
- Load 12,410 chunks from local ChromaDB
- Upload them to your Qdrant Cloud cluster
- Create a collection with proper vector configuration (1536 dimensions, cosine similarity)

**Expected time:** 2-5 minutes (100 points per batch)

### 2.3 Verify Upload

The script will show:
```
✅ Migration successful!

Qdrant Collection: novel_chunks_openrouter
Total Points: 12410
Vector Dimensions: 1536
```

## Step 3: Test Locally (Optional)

Before deploying to HF Spaces, test the Gradio app locally:

### 3.1 Install Gradio

```bash
uv pip install gradio
```

### 3.2 Update .env with Qdrant URL

Make sure `.env` has your actual Qdrant cluster URL:
```bash
QDRANT_URL=https://your-actual-cluster-url.qdrant.io:6333
```

### 3.3 Run Gradio App

```bash
python src/deploy/gradio_app.py
```

### 3.4 Test in Browser

1. Open http://localhost:7860
2. Try example queries:
   - "主角的目标是什么？"
   - "方继藩是谁？"
3. Verify results are returned correctly

## Step 4: Deploy to HuggingFace Spaces

### 4.1 Create a New Space

1. Go to [HuggingFace Spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Fill in:
   - **Space name**: `moxin-novel-query` (or your preferred name)
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: CPU basic (free)
4. Click **"Create Space"**

### 4.2 Upload Files

Upload the following files from `deployment/hf_space/` to your Space:

```
deployment/hf_space/
├── app.py              → Upload as app.py
├── requirements.txt    → Upload as requirements.txt
└── README.md          → Upload as README.md
```

**Via Web UI:**
1. Click "Files" tab in your Space
2. Click "Add file" → "Upload files"
3. Upload the 3 files above

**Via Git (Alternative):**
```bash
cd deployment/hf_space
git init
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/moxin-novel-query
git add .
git commit -m "Initial deployment"
git push origin main
```

### 4.3 Configure Secrets

In your HuggingFace Space settings:

1. Click **"Settings"** tab
2. Scroll to **"Repository secrets"**
3. Add the following secrets:

| Name | Value |
|------|-------|
| `QDRANT_URL` | Your Qdrant cluster URL (e.g., `https://abc123.us-east-1-0.aws.cloud.qdrant.io:6333`) |
| `QDRANT_API_KEY` | `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.PXAHs3rw3AR8yQHIMvR9Hvp5KZll179ao3BEDbtXt64` |
| `QDRANT_COLLECTION` | `novel_chunks_openrouter` |
| `OPENROUTER_API_KEY` | `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` |

### 4.4 Wait for Build

HuggingFace will automatically:
1. Install dependencies from `requirements.txt`
2. Build the Space
3. Launch the Gradio app

**Build time:** 2-3 minutes

### 4.5 Access Your App

Once built, your app will be available at:
```
https://huggingface.co/spaces/YOUR_USERNAME/moxin-novel-query
```

## Step 5: Verify Deployment

1. Open your Space URL
2. Try the example queries
3. Verify search results are returned correctly
4. Test with custom queries

## Troubleshooting

### Issue: "Failed to connect to Qdrant"

**Solution:** Check that:
- `QDRANT_URL` includes the port (`:6333`)
- `QDRANT_API_KEY` is correct
- Your Qdrant cluster is running (check cloud.qdrant.io)

### Issue: "Collection not found"

**Solution:**
- Verify migration completed successfully
- Check collection name matches (`novel_chunks_openrouter`)

### Issue: "OpenRouter API error"

**Solution:**
- Verify `OPENROUTER_API_KEY` is set correctly
- Check you have credits in your OpenRouter account

### Issue: Space build fails

**Solution:**
- Check `requirements.txt` syntax
- Ensure all files are uploaded correctly
- Check Space logs for specific error messages

## Cost Breakdown

### Free Tier Limits:
- **Qdrant Cloud**: 1GB storage (enough for ~1M vectors at 1536 dimensions)
  - Your usage: 12,410 chunks = ~74MB ✅
- **HuggingFace Spaces**: Free CPU basic
- **OpenRouter API**: Pay per use
  - Query embedding: ~$0.00002 per query (very cheap)

### Estimated Monthly Cost:
- Qdrant Cloud: **$0** (free tier)
- HuggingFace Spaces: **$0** (free tier)
- OpenRouter API: **~$0.50/month** (for ~25,000 queries)

**Total: ~$0.50/month** 🎉

## Next Steps

After successful deployment:

1. **Share your Space** - Make it public or share the link
2. **Add more features**:
   - Q&A with LLM integration
   - Chapter summarization
   - Character relationship extraction
3. **Upgrade if needed**:
   - HF Spaces: Upgrade to GPU for faster inference
   - Qdrant: Upgrade for more storage/throughput

## Support

For issues:
- Qdrant docs: https://qdrant.tech/documentation/
- HuggingFace Spaces docs: https://huggingface.co/docs/hub/spaces
- Moxin GitHub: https://github.com/yourusername/moxin

## Summary

✅ **What you've built:**
- Cloud-hosted vector database (Qdrant Cloud)
- Public web interface (HF Spaces)
- Semantic search over 1,819 chapters
- 12,410 indexed chunks with full context

✅ **What users can do:**
- Search the novel with natural language
- Find relevant passages instantly
- Ask questions about characters/plot
- Access from anywhere with internet

Congratulations! 🎉 Your novel query system is now live!

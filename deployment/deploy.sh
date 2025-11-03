#!/bin/bash
# Automated deployment script for Moxin to Qdrant Cloud + HuggingFace Spaces

set -e  # Exit on error

echo "======================================"
echo "  Moxin Deployment Automation Script"
echo "======================================"
echo ""

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "❌ Error: .env file not found!"
    exit 1
fi

# Check required environment variables
if [ -z "$QDRANT_URL" ]; then
    echo "❌ Error: QDRANT_URL not set in .env"
    echo "Please create a Qdrant Cloud cluster and update QDRANT_URL in .env"
    exit 1
fi

if [ -z "$QDRANT_API_KEY" ]; then
    echo "❌ Error: QDRANT_API_KEY not set in .env"
    exit 1
fi

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "❌ Error: OPENROUTER_API_KEY not set in .env"
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: HF_TOKEN not set in .env"
    exit 1
fi

echo "✅ Environment variables loaded"
echo ""

# Step 1: Check if embeddings exist
echo "Step 1: Checking local embeddings..."
if [ ! -d "data/embeddings_openrouter" ]; then
    echo "❌ Error: data/embeddings_openrouter not found"
    echo "Please run indexing first: python src/cli/index_novel.py --model openrouter-small"
    exit 1
fi
echo "✅ Local embeddings found"
echo ""

# Step 2: Install dependencies
echo "Step 2: Installing deployment dependencies..."
source .venv/bin/activate
uv pip install qdrant-client gradio -q
echo "✅ Dependencies installed"
echo ""

# Step 3: Migrate to Qdrant Cloud
echo "Step 3: Migrating embeddings to Qdrant Cloud..."
echo "This may take 2-5 minutes..."
python src/deploy/migrate_to_qdrant.py \
    --chromadb data/embeddings_openrouter \
    --collection novel_chunks_openrouter \
    --qdrant-collection novel_chunks_openrouter \
    --batch-size 100

if [ $? -eq 0 ]; then
    echo "✅ Migration completed successfully"
else
    echo "❌ Migration failed"
    exit 1
fi
echo ""

# Step 4: Test locally (optional)
echo "Step 4: Test deployment locally? (y/n)"
read -r test_local

if [ "$test_local" = "y" ] || [ "$test_local" = "Y" ]; then
    echo "Starting local Gradio server..."
    echo "Press Ctrl+C to stop when done testing"
    python src/deploy/gradio_app.py
    echo ""
fi

# Step 5: Prepare HF Space files
echo "Step 5: Preparing HuggingFace Space files..."
echo "Files ready in deployment/hf_space/"
echo ""

# Step 6: Instructions for HF Spaces
echo "======================================"
echo "  Next Steps: Deploy to HF Spaces"
echo "======================================"
echo ""
echo "1. Go to https://huggingface.co/spaces"
echo "2. Click 'Create new Space'"
echo "3. Choose:"
echo "   - Name: moxin-novel-query"
echo "   - SDK: Gradio"
echo "   - Hardware: CPU basic (free)"
echo ""
echo "4. Upload files from deployment/hf_space/:"
echo "   - app.py"
echo "   - requirements.txt"
echo "   - README.md"
echo ""
echo "5. Add the following secrets in Space settings:"
echo "   QDRANT_URL=$QDRANT_URL"
echo "   QDRANT_API_KEY=$QDRANT_API_KEY"
echo "   QDRANT_COLLECTION=novel_chunks_openrouter"
echo "   OPENROUTER_API_KEY=$OPENROUTER_API_KEY"
echo ""
echo "6. Wait for Space to build (2-3 minutes)"
echo ""
echo "7. Your app will be live at:"
echo "   https://huggingface.co/spaces/YOUR_USERNAME/moxin-novel-query"
echo ""
echo "======================================"
echo "  Deployment Complete! 🎉"
echo "======================================"
echo ""
echo "For detailed instructions, see deployment/DEPLOYMENT_GUIDE.md"

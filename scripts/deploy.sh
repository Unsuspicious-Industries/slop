#!/bin/bash
# Deploy and setup on remote server
# Usage: ./scripts/deploy.sh user@server:/path/to/deploy

set -e

REMOTE=$1
if [ -z "$REMOTE" ]; then
    echo "Usage: $0 user@server:/path/to/deploy"
    exit 1
fi

echo "Deploying to $REMOTE"

# Create deployment archive
echo "Creating deployment archive..."
tar -czf slop-deploy.tar.gz \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='.venv' \
    --exclude='*.pyc' \
    --exclude='outputs/*' \
    --exclude='data/generated/*' \
    --exclude='.DS_Store' \
    .

# Copy to remote
echo "Copying to remote server..."
scp slop-deploy.tar.gz $REMOTE/

# Extract and setup on remote
REMOTE_HOST=$(echo $REMOTE | cut -d: -f1)
REMOTE_PATH=$(echo $REMOTE | cut -d: -f2)

echo "Setting up on remote server..."
ssh $REMOTE_HOST << EOF
    set -e
    cd $REMOTE_PATH
    tar -xzf slop-deploy.tar.gz
    rm slop-deploy.tar.gz
    
    echo "Setting up Python environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    
    echo "Deployment complete!"
    echo "Run: ssh $REMOTE_HOST 'cd $REMOTE_PATH && source .venv/bin/activate && python scripts/run_full_analysis.py'"
EOF

# Cleanup local archive
rm slop-deploy.tar.gz

echo "Deployment successful!"

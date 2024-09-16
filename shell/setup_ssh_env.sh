#!/bin/bash

# Start the SSH agent
eval "$(ssh-agent -s)"

# Add your SSH private key to the agent (ensure the path to your key is correct)
ssh-add ~/.ssh/id_rsa

# Test the SSH connection to GitHub
ssh -T git@github.com

# Ensure the submodule uses SSH instead of HTTPS (modify the path if necessary)
cd /workspaces/luma
git config --file .gitmodules submodule.luma/neural.url git@github.com:ChanLumerico/luma-neural.git

cd /workspaces/luma/luma/neural
git config --file .gitmodules submodule.autoprop.url git@github.com:ChanLumerico/autoprop.git

cd /workspaces/luma

# Sync and initialize submodules (if needed)
git submodule sync
git submodule update --init --recursive

cd luma/neural
git submodule sync
git submodule update --init --recursive

cd /workspaces/luma

# Optional: Show the current Git remote URLs to confirm SSH is in use
echo "Git remote URL for luma:"
git remote -v
echo

echo "Git remote URL for submodule (luma-neural):"
cd luma/neural
git remote -v
echo

echo "Git remote URL for submodule (autoprop):"
cd autoprop
git remote -v
echo

echo "Dry push for submodule (luma-neural)"
cd /workspaces/luma/luma/neural
git push --dry-run
echo

echo "Dry push for submodule (autoprop)"
cd autoprop
git push --dry-run
echo

# Now you're ready to commit and push changes
echo "Setup complete! You can now commit and push changes to submodules or the main repo."

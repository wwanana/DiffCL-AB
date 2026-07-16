#!/usr/bin/env bash
set -euo pipefail

# Push the prepared public release to GitHub.
#
# Usage:
#   bash push_to_github.sh <github_repo_url> "<git_user_name>" "<git_user_email>"
#
# Examples:
#   bash push_to_github.sh \
#     https://github.com/YOUR_USERNAME/DiffCL-AB.git \
#     "Your Name" \
#     "your_email@example.com"
#
#   bash push_to_github.sh \
#     git@github.com:YOUR_USERNAME/DiffCL-AB.git \
#     "Your Name" \
#     "your_email@example.com"

REPO_URL="${1:-}"
GIT_USER_NAME="${2:-}"
GIT_USER_EMAIL="${3:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -z "$REPO_URL" || -z "$GIT_USER_NAME" || -z "$GIT_USER_EMAIL" ]]; then
  echo "Usage:"
  echo "  bash push_to_github.sh <github_repo_url> \"<git_user_name>\" \"<git_user_email>\""
  exit 2
fi

case "$REPO_URL" in
  https://github.com/*.git|git@github.com:*.git)
    ;;
  *)
    echo "ERROR: Invalid GitHub repository URL: $REPO_URL"
    echo "Expected HTTPS: https://github.com/USERNAME/REPOSITORY.git"
    echo "or SSH:         git@github.com:USERNAME/REPOSITORY.git"
    exit 2
    ;;
esac

echo "============================================================"
echo "Repository root: $ROOT"
echo "GitHub remote:   $REPO_URL"
echo "Git author:      $GIT_USER_NAME <$GIT_USER_EMAIL>"
echo "============================================================"

required_files=("README.md" "model.py" "lit_model.py" "dataset.py")
for f in "${required_files[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: Required public-release file is missing: $f"
    exit 3
  fi
done

echo "[1/7] Updating .gitignore..."
touch .gitignore
ignore_patterns=(
  "__pycache__/"
  "*.py[cod]"
  ".pytest_cache/"
  ".ipynb_checkpoints/"
  ".DS_Store"
  "*.log"
  "lightning_logs/"
  "wandb/"
  "checkpoints*/"
  "cv5_results*/"
  "results*/"
  "result_log/"
  "*.ckpt"
  "*.pt"
  "*.pth"
  "*.bin"
  "*.safetensors"
  "model/esm2_650m"
  "model/esm2_650m/"
  "data/**/*.csv"
  ".env"
  "*.pem"
  "*.key"
)

for pattern in "${ignore_patterns[@]}"; do
  grep -qxF "$pattern" .gitignore || echo "$pattern" >> .gitignore
done

echo "[2/7] Checking for secrets and server-specific paths..."
SECRET_HITS="$(
  grep -RInE \
    'ghp_[A-Za-z0-9]+|github_pat_[A-Za-z0-9_]+|AKIA[0-9A-Z]{16}|BEGIN (RSA |OPENSSH )?PRIVATE KEY|password[[:space:]]*=|secret[[:space:]]*=|token[[:space:]]*=' \
    . \
    --exclude-dir=.git \
    --exclude=push_to_github.sh \
    2>/dev/null || true
)"

if [[ -n "$SECRET_HITS" ]]; then
  echo "ERROR: Possible secret material was found:"
  echo "$SECRET_HITS"
  echo "Remove the secret before publishing."
  exit 4
fi

ROOT_PATH_HITS="$(
  grep -RIn '/root/autodl-tmp/' \
    . \
    --exclude-dir=.git \
    --exclude=push_to_github.sh \
    --exclude='*.md' \
    2>/dev/null || true
)"

if [[ -n "$ROOT_PATH_HITS" ]]; then
  echo "WARNING: Hard-coded AutoDL paths were found in code/config files:"
  echo "$ROOT_PATH_HITS"
  echo
  read -r -p "Continue anyway? [y/N] " answer
  [[ "$answer" =~ ^[Yy]$ ]] || exit 5
fi

echo "[3/7] Checking forbidden weights, checkpoints, and large files..."
FORBIDDEN="$(
  find . \
    -path './.git' -prune -o \
    -type f \
    \( -iname '*.ckpt' -o -iname '*.pt' -o -iname '*.pth' -o \
       -iname '*.bin' -o -iname '*.safetensors' \) \
    -print
)"

if [[ -n "$FORBIDDEN" ]]; then
  echo "ERROR: Model/checkpoint files are still present:"
  echo "$FORBIDDEN"
  echo "Move or delete them before publishing."
  exit 6
fi

LARGE_FILES="$(
  find . \
    -path './.git' -prune -o \
    -type f -size +50M -print
)"

if [[ -n "$LARGE_FILES" ]]; then
  echo "ERROR: Files larger than 50 MB were found:"
  echo "$LARGE_FILES"
  echo "Remove them or use an external release/data host."
  exit 7
fi

echo "[4/7] Initializing Git..."
if [[ ! -d .git ]]; then
  git init
fi

git config user.name "$GIT_USER_NAME"
git config user.email "$GIT_USER_EMAIL"
git branch -M main

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi

echo "[5/7] Staging public files..."
git add .

STAGED_FORBIDDEN="$(
  git diff --cached --name-only \
    | grep -Ei '\.(ckpt|pt|pth|bin|safetensors)$' || true
)"
if [[ -n "$STAGED_FORBIDDEN" ]]; then
  echo "ERROR: Forbidden files were staged:"
  echo "$STAGED_FORBIDDEN"
  exit 8
fi

echo
echo "Files ready to publish:"
git status --short
echo

read -r -p "Commit and push these files to GitHub? [y/N] " confirm
[[ "$confirm" =~ ^[Yy]$ ]] || {
  echo "Cancelled before commit."
  exit 0
}

echo "[6/7] Creating commit..."
if git diff --cached --quiet; then
  echo "No new staged changes. Existing commit history will be pushed."
else
  git commit -m "Initial public release"
fi

echo "[7/7] Pushing main branch..."
git push -u origin main

echo
echo "============================================================"
echo "SUCCESS: project pushed to GitHub."
echo "Remote: $REPO_URL"
echo "============================================================"

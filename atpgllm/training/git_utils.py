import subprocess
import os
import wandb

def get_git_info():
    """Extracts Git repository name, branch, and commit hash."""
    try:
        repo = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url'], stderr=subprocess.DEVNULL).decode('utf-8').strip()
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.DEVNULL).decode('utf-8').strip()
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('utf-8').strip()
        return {"git_repo": repo, "git_branch": branch, "git_commit": commit}
    except Exception:
        # Fails silently if not a git repo or git is not installed
        return {}
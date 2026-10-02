import subprocess
import os
import wandb

def get_git_info(root=None, prefix="git_"):
    """Extracts Git repository name, branch, commit hash and dirty flag of ``root`` (default: cwd)."""
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root or '.'), *args], stderr=subprocess.DEVNULL).decode('utf-8').strip()
    try:
        return {f"{prefix}repo": git('config', '--get', 'remote.origin.url'),
                f"{prefix}branch": git('rev-parse', '--abbrev-ref', 'HEAD'),
                f"{prefix}commit": git('rev-parse', 'HEAD'),
                f"{prefix}dirty": bool(git('status', '--porcelain', '--untracked-files=no'))}
    except Exception:
        # Fails silently if not a git repo or git is not installed
        return {}

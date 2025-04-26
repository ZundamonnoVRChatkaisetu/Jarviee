"""
Code Repository Manager for Jarviee

This module provides an interface for working with code repositories,
including version control systems like Git, as well as code hosting
platforms like GitHub, GitLab, and Bitbucket.
"""

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import git
from git import Repo


class CodeRepository:
    """
    Interface for working with code repositories.
    
    This class provides functionality for interacting with version control
    systems and code hosting platforms, making it easy to manage code
    repositories as part of Jarviee's programming capabilities.
    """
    
    def __init__(self, repo_path: Optional[str] = None):
        """
        Initialize the code repository manager.
        
        Args:
            repo_path: Path to the repository on disk (if None, must be set later)
        """
        self.logger = logging.getLogger("code_repository")
        self.repo_path = repo_path
        self.repo: Optional[Repo] = None
        
        if repo_path:
            self.set_repo_path(repo_path)
    
    def set_repo_path(self, repo_path: str) -> bool:
        """
        Set the repository path and initialize the Git repository object.
        
        Args:
            repo_path: Path to the repository on disk
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.repo_path = repo_path
            
            # Check if the path exists
            if not os.path.exists(repo_path):
                self.logger.error(f"Repository path does not exist: {repo_path}")
                return False
            
            # Initialize the repository object
            self.repo = Repo(repo_path)
            return True
        
        except git.exc.InvalidGitRepositoryError:
            self.logger.error(f"Not a valid Git repository: {repo_path}")
            return False
        
        except Exception as e:
            self.logger.error(f"Error initializing repository: {str(e)}")
            return False
    
    def create_repository(
        self, 
        path: str, 
        bare: bool = False,
        initial_branch: str = "main"
    ) -> bool:
        """
        Create a new Git repository.
        
        Args:
            path: Path where to create the repository
            bare: Whether to create a bare repository
            initial_branch: Name of the initial branch
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create the repository
            self.repo = Repo.init(path, bare=bare, initial_branch=initial_branch)
            self.repo_path = path
            return True
        
        except Exception as e:
            self.logger.error(f"Error creating repository: {str(e)}")
            return False
    
    def clone_repository(
        self, 
        url: str, 
        path: str,
        branch: Optional[str] = None,
        depth: Optional[int] = None
    ) -> bool:
        """
        Clone a Git repository.
        
        Args:
            url: URL of the repository to clone
            path: Path where to clone the repository
            branch: Branch to clone (if None, clones the default branch)
            depth: Depth of history to clone (if None, clones the full history)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare the clone options
            kwargs = {}
            if branch:
                kwargs["branch"] = branch
            if depth:
                kwargs["depth"] = depth
            
            # Clone the repository
            self.repo = Repo.clone_from(url, path, **kwargs)
            self.repo_path = path
            return True
        
        except Exception as e:
            self.logger.error(f"Error cloning repository: {str(e)}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the repository.
        
        Returns:
            Dictionary with repository status information
        """
        if not self.repo:
            return {"error": "Repository not initialized"}
        
        try:
            # Get the repository status
            status = {
                "branch": self.repo.active_branch.name,
                "head": self.repo.head.commit.hexsha,
                "dirty": self.repo.is_dirty(),
                "untracked_files": self.repo.untracked_files,
                "staged_files": [],
                "unstaged_files": [],
                "branches": [b.name for b in self.repo.branches],
                "remotes": [r.name for r in self.repo.remotes]
            }
            
            # Get staged and unstaged files
            for item in self.repo.index.diff("HEAD"):
                status["staged_files"].append(item.a_path)
            
            for item in self.repo.index.diff(None):
                status["unstaged_files"].append(item.a_path)
            
            return status
        
        except Exception as e:
            self.logger.error(f"Error getting repository status: {str(e)}")
            return {"error": str(e)}
    
    def get_branches(self) -> List[Dict[str, Any]]:
        """
        Get a list of branches in the repository.
        
        Returns:
            List of dictionaries with branch information
        """
        if not self.repo:
            return [{"error": "Repository not initialized"}]
        
        try:
            # Get the branches
            branches = []
            
            for branch in self.repo.branches:
                branch_info = {
                    "name": branch.name,
                    "commit": branch.commit.hexsha,
                    "active": branch.name == self.repo.active_branch.name,
                    "upstream": None
                }
                
                # Try to get upstream branch
                try:
                    tracking_branch = branch.tracking_branch()
                    if tracking_branch:
                        branch_info["upstream"] = tracking_branch.name
                except:
                    pass
                
                branches.append(branch_info)
            
            return branches
        
        except Exception as e:
            self.logger.error(f"Error getting branches: {str(e)}")
            return [{"error": str(e)}]
    
    def create_branch(
        self, 
        name: str, 
        start_point: Optional[str] = None,
        checkout: bool = False
    ) -> bool:
        """
        Create a new branch.
        
        Args:
            name: Name of the branch to create
            start_point: Revision to start the branch from (if None, starts from HEAD)
            checkout: Whether to checkout the new branch
            
        Returns:
            True if successful, False otherwise
        """
        if not self.repo:
            self.logger.error("Repository not initialized")
            return False
        
        try:
            # Create the branch
            if start_point:
                self.repo.git.branch(name, start_point)
            else:
                self.repo.git.branch(name)
            
            # Checkout the branch if requested
            if checkout:
                self.repo.git.checkout(name)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error creating branch: {str(e)}")
            return False
    
    def checkout_branch(self, name: str, create: bool = False) -> bool:
        """
        Checkout a branch.
        
        Args:
            name: Name of the branch to checkout
            create: Whether to create the branch if it doesn't exist
            
        Returns:
            True if successful, False otherwise
        """
        if not self.repo:
            self.logger.error("Repository not initialized")
            return False
        
        try:
            # Checkout the branch
            if create:
                self.repo.git.checkout("-b", name)
            else:
                self.repo.git.checkout(name)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error checking out branch: {str(e)}")
            return False
    
    def get_commits(
        self, 
        max_count: int = 10, 
        skip: int = 0,
        branch: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get a list of commits from the repository.
        
        Args:
            max_count: Maximum number of commits to retrieve
            skip: Number of commits to skip
            branch: Branch to get commits from (if None, uses the active branch)
            
        Returns:
            List of dictionaries with commit information
        """
        if not self.repo:
            return [{"error": "Repository not initialized"}]
        
        try:
            # Get the commits
            target = branch if branch else self.repo.active_branch.name
            commits = []
            
            for commit in self.repo.iter_commits(target, max_count=max_count, skip=skip):
                commit_info = {
                    "hash": commit.hexsha,
                    "short_hash": commit.hexsha[:7],
                    "message": commit.message.strip(),
                    "author": {
                        "name": commit.author.name,
                        "email": commit.author.email
                    },
                    "committer": {
                        "name": commit.committer.name,
                        "email": commit.committer.email
                    },
                    "authored_date": commit.authored_datetime.isoformat(),
                    "committed_date": commit.committed_datetime.isoformat(),
                    "parents": [p.hexsha for p in commit.parents]
                }
                
                commits.append(commit_info)
            
            return commits
        
        except Exception as e:
            self.logger.error(f"Error getting commits: {str(e)}")
            return [{"error": str(e)}]
    
    def get_commit(self, commit_hash: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific commit.
        
        Args:
            commit_hash: Hash of the commit to retrieve
            
        Returns:
            Dictionary with commit information
        """
        if not self.repo:
            return {"error": "Repository not initialized"}
        
        try:
            # Get the commit
            commit = self.repo.commit(commit_hash)
            
            # Get the commit stats
            stats = commit.stats
            
            # Get the commit diff
            diffs = []
            for diff in commit.diff(commit.parents[0] if commit.parents else git.NULL_TREE):
                diff_info = {
                    "a_path": diff.a_path,
                    "b_path": diff.b_path,
                    "change_type": diff.change_type,
                    "renamed": diff.renamed,
                    "deleted": diff.deleted,
                    "new_file": diff.new_file
                }
                
                # Try to get the diff content
                try:
                    diff_info["diff"] = diff.diff.decode("utf-8")
                except:
                    diff_info["diff"] = "Binary file or encoding error"
                
                diffs.append(diff_info)
            
            # Build the commit info
            commit_info = {
                "hash": commit.hexsha,
                "short_hash": commit.hexsha[:7],
                "message": commit.message.strip(),
                "author": {
                    "name": commit.author.name,
                    "email": commit.author.email
                },
                "committer": {
                    "name": commit.committer.name,
                    "email": commit.committer.email
                },
                "authored_date": commit.authored_datetime.isoformat(),
                "committed_date": commit.committed_datetime.isoformat(),
                "parents": [p.hexsha for p in commit.parents],
                "stats": {
                    "files": len(stats["files"]),
                    "insertions": stats["total"]["insertions"],
                    "deletions": stats["total"]["deletions"],
                    "lines": stats["total"]["lines"],
                    "files_details": stats["files"]
                },
                "diffs": diffs
            }
            
            return commit_info
        
        except Exception as e:
            self.logger.error(f"Error getting commit: {str(e)}")
            return {"error": str(e)}
    
    def add_files(self, paths: List[str]) -> bool:
        """
        Add files to the staging area.
        
        Args:
            paths: List of file paths to add
            
        Returns:
            True if successful, False otherwise
        """
        if not self.repo:
            self.logger.error("Repository not initialized")
            return False
        
        try:
            # Add the files
            if paths:
                self.repo.git.add(paths)
            else:
                self.repo.git.add(".")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error adding files: {str(e)}")
            return False
    
    def commit(
        self, 
        message: str, 
        author: Optional[str] = None,
        author_email: Optional[str] = None
    ) -> bool:
        """
        Commit changes to the repository.
        
        Args:
            message: Commit message
            author: Author name (if None, uses the configured author)
            author_email: Author email (if None, uses the configured email)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.repo:
            self.logger.error("Repository not initialized")
            return False
        
        try:
            # Prepare the commit options
            kwargs = {}
            if author and author_email:
                kwargs["author"] = f"{author} <{author_email}>"
            
            # Commit the changes
            self.repo.git.commit("-m", message, **kwargs)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error committing changes: {str(e)}")
            return False
    
    def push(
        self, 
        remote: str = "origin", 
        branch: Optional[str] = None,
        set_upstream: bool = False
    ) -> bool:
        """
        Push changes to a remote repository.
        
        Args:
            remote: Name of the remote repository
            branch: Branch to push (if None, pushes the active branch)
            set_upstream: Whether to set the upstream branch
            
        Returns:
            True if successful, False otherwise
        """
        if not self.repo:
            self.logger.error("Repository not initialized")
            return False
        
        try:
            # Get the branch name
            target_branch = branch if branch else self.repo.active_branch.name
            
            # Push the changes
            if set_upstream:
                self.repo.git.push("-u", remote, target_branch)
            else:
                self.repo.git.push(remote, target_branch)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error pushing changes: {str(e)}")
            return False
    
    def pull(
        self, 
        remote: str = "origin", 
        branch: Optional[str] = None,
        rebase: bool = False
    ) -> bool:
        """
        Pull changes from a remote repository.
        
        Args:
            remote: Name of the remote repository
            branch: Branch to pull (if None, pulls the active branch)
            rebase: Whether to rebase instead of merge
            
        Returns:
            True if successful, False otherwise
        """
        if not self.repo:
            self.logger.error("Repository not initialized")
            return False
        
        try:
            # Get the branch name
            target_branch = branch if branch else self.repo.active_branch.name
            
            # Pull the changes
            if rebase:
                self.repo.git.pull("--rebase", remote, target_branch)
            else:
                self.repo.git.pull(remote, target_branch)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error pulling changes: {str(e)}")
            return False
    
    def get_file_history(
        self, 
        file_path: str, 
        max_count: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get the commit history for a specific file.
        
        Args:
            file_path: Path to the file
            max_count: Maximum number of commits to retrieve
            
        Returns:
            List of dictionaries with commit information
        """
        if not self.repo:
            return [{"error": "Repository not initialized"}]
        
        try:
            # Get the file history
            if not os.path.exists(os.path.join(self.repo_path, file_path)):
                return [{"error": f"File not found: {file_path}"}]
            
            commits = []
            
            for commit in self.repo.iter_commits(paths=file_path, max_count=max_count):
                commit_info = {
                    "hash": commit.hexsha,
                    "short_hash": commit.hexsha[:7],
                    "message": commit.message.strip(),
                    "author": {
                        "name": commit.author.name,
                        "email": commit.author.email
                    },
                    "authored_date": commit.authored_datetime.isoformat()
                }
                
                commits.append(commit_info)
            
            return commits
        
        except Exception as e:
            self.logger.error(f"Error getting file history: {str(e)}")
            return [{"error": str(e)}]
    
    def get_file_content(
        self, 
        file_path: str, 
        revision: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get the content of a file in the repository.
        
        Args:
            file_path: Path to the file
            revision: Revision to get the file from (if None, gets from HEAD)
            
        Returns:
            Dictionary with file content information
        """
        if not self.repo:
            return {"error": "Repository not initialized"}
        
        try:
            # Get the file content
            target = revision if revision else "HEAD"
            
            try:
                content = self.repo.git.show(f"{target}:{file_path}")
                return {
                    "path": file_path,
                    "content": content,
                    "revision": target,
                    "exists": True
                }
            except git.exc.GitCommandError:
                # File might not exist in the specified revision
                return {
                    "path": file_path,
                    "content": None,
                    "revision": target,
                    "exists": False
                }
        
        except Exception as e:
            self.logger.error(f"Error getting file content: {str(e)}")
            return {"error": str(e)}
    
    def get_diff(
        self, 
        file_path: Optional[str] = None,
        staged: bool = False
    ) -> str:
        """
        Get the diff for a file or the entire repository.
        
        Args:
            file_path: Path to the file (if None, gets diff for all files)
            staged: Whether to get the diff for staged changes
            
        Returns:
            Diff as a string
        """
        if not self.repo:
            return "Error: Repository not initialized"
        
        try:
            # Get the diff
            if staged:
                if file_path:
                    return self.repo.git.diff("--staged", file_path)
                else:
                    return self.repo.git.diff("--staged")
            else:
                if file_path:
                    return self.repo.git.diff(file_path)
                else:
                    return self.repo.git.diff()
        
        except Exception as e:
            self.logger.error(f"Error getting diff: {str(e)}")
            return f"Error: {str(e)}"
    
    def get_blame(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Get blame information for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of dictionaries with blame information for each line
        """
        if not self.repo:
            return [{"error": "Repository not initialized"}]
        
        try:
            # Get the blame information
            blame_lines = []
            
            # Get the blame output
            blame_output = self.repo.git.blame(file_path, "--porcelain")
            
            # Parse the blame output
            commit = None
            author = None
            author_email = None
            author_time = None
            commit_time = None
            line_content = None
            
            for line in blame_output.split("\n"):
                if line.startswith("author "):
                    author = line[7:]
                elif line.startswith("author-mail "):
                    author_email = line[13:].strip("<>")
                elif line.startswith("author-time "):
                    author_time = line[12:]
                elif line.startswith("committer-time "):
                    commit_time = line[15:]
                elif line.startswith("\t"):
                    # Line content
                    line_content = line[1:]
                    
                    blame_lines.append({
                        "commit": commit,
                        "author": author,
                        "author_email": author_email,
                        "author_time": author_time,
                        "commit_time": commit_time,
                        "content": line_content
                    })
                    
                    # Reset for the next line
                    commit = None
                    author = None
                    author_email = None
                    author_time = None
                    commit_time = None
                    line_content = None
                else:
                    # Commit hash
                    parts = line.split(" ")
                    if len(parts) >= 4:
                        commit = parts[0]
            
            return blame_lines
        
        except Exception as e:
            self.logger.error(f"Error getting blame: {str(e)}")
            return [{"error": str(e)}]
    
    def create_tag(
        self, 
        tag_name: str, 
        message: Optional[str] = None,
        commit: Optional[str] = None
    ) -> bool:
        """
        Create a new tag.
        
        Args:
            tag_name: Name of the tag to create
            message: Tag message (if None, creates a lightweight tag)
            commit: Commit to tag (if None, tags HEAD)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.repo:
            self.logger.error("Repository not initialized")
            return False
        
        try:
            # Create the tag
            if message:
                self.repo.create_tag(tag_name, message=message, ref=commit)
            else:
                self.repo.create_tag(tag_name, ref=commit)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error creating tag: {str(e)}")
            return False
    
    def get_tags(self) -> List[Dict[str, Any]]:
        """
        Get a list of tags in the repository.
        
        Returns:
            List of dictionaries with tag information
        """
        if not self.repo:
            return [{"error": "Repository not initialized"}]
        
        try:
            # Get the tags
            tags = []
            
            for tag in self.repo.tags:
                tag_info = {
                    "name": tag.name,
                    "commit": tag.commit.hexsha,
                    "message": tag.tag.message if hasattr(tag, "tag") and tag.tag else None,
                    "date": tag.commit.committed_datetime.isoformat()
                }
                
                tags.append(tag_info)
            
            return tags
        
        except Exception as e:
            self.logger.error(f"Error getting tags: {str(e)}")
            return [{"error": str(e)}]
    
    def reset(
        self, 
        path: Optional[str] = None,
        hard: bool = False
    ) -> bool:
        """
        Reset changes in the repository.
        
        Args:
            path: Path to reset (if None, resets the entire repository)
            hard: Whether to perform a hard reset (discards all changes)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.repo:
            self.logger.error("Repository not initialized")
            return False
        
        try:
            # Reset the changes
            if hard:
                if path:
                    self.repo.git.checkout("HEAD", "--", path)
                else:
                    self.repo.git.reset("--hard")
            else:
                if path:
                    self.repo.git.reset("HEAD", "--", path)
                else:
                    self.repo.git.reset()
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error resetting changes: {str(e)}")
            return False


# GitHub specific functionality

class GitHubRepository:
    """
    Interface for working with GitHub repositories.
    
    This class provides functionality for interacting with GitHub repositories,
    such as creating issues, pull requests, and managing repository settings.
    """
    
    def __init__(
        self, 
        repo_path: Optional[str] = None,
        repo_url: Optional[str] = None,
        token: Optional[str] = None
    ):
        """
        Initialize the GitHub repository manager.
        
        Args:
            repo_path: Path to the repository on disk
            repo_url: URL of the GitHub repository
            token: GitHub API token
        """
        self.logger = logging.getLogger("github_repository")
        self.code_repo = CodeRepository(repo_path) if repo_path else None
        self.repo_url = repo_url
        self.token = token
        
        # Extract owner and repo name from URL if provided
        self.owner = None
        self.repo_name = None
        
        if repo_url:
            self._parse_repo_url(repo_url)
    
    def _parse_repo_url(self, url: str) -> None:
        """
        Parse a GitHub repository URL to extract owner and repo name.
        
        Args:
            url: GitHub repository URL
        """
        patterns = [
            r"https://github\.com/([^/]+)/([^/.]+)",
            r"git@github\.com:([^/]+)/([^/.]+)",
            r"ssh://git@github\.com/([^/]+)/([^/.]+)"
        ]
        
        for pattern in patterns:
            match = re.match(pattern, url)
            if match:
                self.owner = match.group(1)
                self.repo_name = match.group(2)
                return
    
    def set_repo_path(self, repo_path: str) -> bool:
        """
        Set the repository path and initialize the Git repository object.
        
        Args:
            repo_path: Path to the repository on disk
            
        Returns:
            True if successful, False otherwise
        """
        # Create a new CodeRepository or update the existing one
        if self.code_repo:
            return self.code_repo.set_repo_path(repo_path)
        else:
            self.code_repo = CodeRepository(repo_path)
            return self.code_repo.repo is not None
    
    def set_repo_url(self, url: str) -> None:
        """
        Set the GitHub repository URL.
        
        Args:
            url: GitHub repository URL
        """
        self.repo_url = url
        self._parse_repo_url(url)
    
    def set_token(self, token: str) -> None:
        """
        Set the GitHub API token.
        
        Args:
            token: GitHub API token
        """
        self.token = token
    
    def create_issue(
        self, 
        title: str, 
        body: str, 
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a new issue in the GitHub repository.
        
        Args:
            title: Issue title
            body: Issue body
            labels: List of labels to apply
            assignees: List of users to assign
            
        Returns:
            Dictionary with issue information
        """
        if not self.owner or not self.repo_name:
            return {"error": "Repository owner and name not set"}
        
        if not self.token:
            return {"error": "GitHub API token not set"}
        
        try:
            # Prepare the API request
            import requests
            
            headers = {
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            data = {
                "title": title,
                "body": body
            }
            
            if labels:
                data["labels"] = labels
            
            if assignees:
                data["assignees"] = assignees
            
            # Send the request
            response = requests.post(
                f"https://api.github.com/repos/{self.owner}/{self.repo_name}/issues",
                headers=headers,
                json=data
            )
            
            # Check the response
            if response.status_code == 201:
                return response.json()
            else:
                return {
                    "error": f"Failed to create issue: {response.status_code}",
                    "details": response.json()
                }
        
        except Exception as e:
            self.logger.error(f"Error creating issue: {str(e)}")
            return {"error": str(e)}
    
    def create_pull_request(
        self, 
        title: str, 
        body: str, 
        base: str, 
        head: str,
        draft: bool = False
    ) -> Dict[str, Any]:
        """
        Create a new pull request in the GitHub repository.
        
        Args:
            title: Pull request title
            body: Pull request body
            base: Branch to merge into
            head: Branch with changes
            draft: Whether to create a draft pull request
            
        Returns:
            Dictionary with pull request information
        """
        if not self.owner or not self.repo_name:
            return {"error": "Repository owner and name not set"}
        
        if not self.token:
            return {"error": "GitHub API token not set"}
        
        try:
            # Prepare the API request
            import requests
            
            headers = {
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            data = {
                "title": title,
                "body": body,
                "base": base,
                "head": head,
                "draft": draft
            }
            
            # Send the request
            response = requests.post(
                f"https://api.github.com/repos/{self.owner}/{self.repo_name}/pulls",
                headers=headers,
                json=data
            )
            
            # Check the response
            if response.status_code == 201:
                return response.json()
            else:
                return {
                    "error": f"Failed to create pull request: {response.status_code}",
                    "details": response.json()
                }
        
        except Exception as e:
            self.logger.error(f"Error creating pull request: {str(e)}")
            return {"error": str(e)}
    
    def get_pull_requests(
        self, 
        state: str = "open",
        base: Optional[str] = None,
        head: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get a list of pull requests in the GitHub repository.
        
        Args:
            state: State of the pull requests to retrieve ('open', 'closed', 'all')
            base: Filter by base branch
            head: Filter by head branch
            
        Returns:
            List of dictionaries with pull request information
        """
        if not self.owner or not self.repo_name:
            return [{"error": "Repository owner and name not set"}]
        
        if not self.token:
            return [{"error": "GitHub API token not set"}]
        
        try:
            # Prepare the API request
            import requests
            
            headers = {
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            params = {
                "state": state
            }
            
            if base:
                params["base"] = base
            
            if head:
                params["head"] = head
            
            # Send the request
            response = requests.get(
                f"https://api.github.com/repos/{self.owner}/{self.repo_name}/pulls",
                headers=headers,
                params=params
            )
            
            # Check the response
            if response.status_code == 200:
                return response.json()
            else:
                return [{
                    "error": f"Failed to get pull requests: {response.status_code}",
                    "details": response.json()
                }]
        
        except Exception as e:
            self.logger.error(f"Error getting pull requests: {str(e)}")
            return [{"error": str(e)}]
    
    def get_issues(
        self, 
        state: str = "open",
        labels: Optional[List[str]] = None,
        assignee: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get a list of issues in the GitHub repository.
        
        Args:
            state: State of the issues to retrieve ('open', 'closed', 'all')
            labels: Filter by labels
            assignee: Filter by assignee
            
        Returns:
            List of dictionaries with issue information
        """
        if not self.owner or not self.repo_name:
            return [{"error": "Repository owner and name not set"}]
        
        if not self.token:
            return [{"error": "GitHub API token not set"}]
        
        try:
            # Prepare the API request
            import requests
            
            headers = {
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            params = {
                "state": state
            }
            
            if labels:
                params["labels"] = ",".join(labels)
            
            if assignee:
                params["assignee"] = assignee
            
            # Send the request
            response = requests.get(
                f"https://api.github.com/repos/{self.owner}/{self.repo_name}/issues",
                headers=headers,
                params=params
            )
            
            # Check the response
            if response.status_code == 200:
                return response.json()
            else:
                return [{
                    "error": f"Failed to get issues: {response.status_code}",
                    "details": response.json()
                }]
        
        except Exception as e:
            self.logger.error(f"Error getting issues: {str(e)}")
            return [{"error": str(e)}]
    
    def get_collaborators(self) -> List[Dict[str, Any]]:
        """
        Get a list of collaborators on the GitHub repository.
        
        Returns:
            List of dictionaries with collaborator information
        """
        if not self.owner or not self.repo_name:
            return [{"error": "Repository owner and name not set"}]
        
        if not self.token:
            return [{"error": "GitHub API token not set"}]
        
        try:
            # Prepare the API request
            import requests
            
            headers = {
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            # Send the request
            response = requests.get(
                f"https://api.github.com/repos/{self.owner}/{self.repo_name}/collaborators",
                headers=headers
            )
            
            # Check the response
            if response.status_code == 200:
                return response.json()
            else:
                return [{
                    "error": f"Failed to get collaborators: {response.status_code}",
                    "details": response.json()
                }]
        
        except Exception as e:
            self.logger.error(f"Error getting collaborators: {str(e)}")
            return [{"error": str(e)}]
    
    def get_repository_info(self) -> Dict[str, Any]:
        """
        Get information about the GitHub repository.
        
        Returns:
            Dictionary with repository information
        """
        if not self.owner or not self.repo_name:
            return {"error": "Repository owner and name not set"}
        
        if not self.token:
            return {"error": "GitHub API token not set"}
        
        try:
            # Prepare the API request
            import requests
            
            headers = {
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            # Send the request
            response = requests.get(
                f"https://api.github.com/repos/{self.owner}/{self.repo_name}",
                headers=headers
            )
            
            # Check the response
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"Failed to get repository info: {response.status_code}",
                    "details": response.json()
                }
        
        except Exception as e:
            self.logger.error(f"Error getting repository info: {str(e)}")
            return {"error": str(e)}
    
    def create_comment(
        self, 
        issue_number: int, 
        body: str
    ) -> Dict[str, Any]:
        """
        Create a comment on an issue or pull request.
        
        Args:
            issue_number: Issue or pull request number
            body: Comment body
            
        Returns:
            Dictionary with comment information
        """
        if not self.owner or not self.repo_name:
            return {"error": "Repository owner and name not set"}
        
        if not self.token:
            return {"error": "GitHub API token not set"}
        
        try:
            # Prepare the API request
            import requests
            
            headers = {
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            data = {
                "body": body
            }
            
            # Send the request
            response = requests.post(
                f"https://api.github.com/repos/{self.owner}/{self.repo_name}/issues/{issue_number}/comments",
                headers=headers,
                json=data
            )
            
            # Check the response
            if response.status_code == 201:
                return response.json()
            else:
                return {
                    "error": f"Failed to create comment: {response.status_code}",
                    "details": response.json()
                }
        
        except Exception as e:
            self.logger.error(f"Error creating comment: {str(e)}")
            return {"error": str(e)}
    
    def merge_pull_request(
        self, 
        pull_number: int,
        merge_method: str = "merge",
        commit_title: Optional[str] = None,
        commit_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Merge a pull request.
        
        Args:
            pull_number: Pull request number
            merge_method: Merge method ('merge', 'squash', 'rebase')
            commit_title: Title for the merge commit
            commit_message: Message for the merge commit
            
        Returns:
            Dictionary with merge result information
        """
        if not self.owner or not self.repo_name:
            return {"error": "Repository owner and name not set"}
        
        if not self.token:
            return {"error": "GitHub API token not set"}
        
        try:
            # Prepare the API request
            import requests
            
            headers = {
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            data = {
                "merge_method": merge_method
            }
            
            if commit_title:
                data["commit_title"] = commit_title
            
            if commit_message:
                data["commit_message"] = commit_message
            
            # Send the request
            response = requests.put(
                f"https://api.github.com/repos/{self.owner}/{self.repo_name}/pulls/{pull_number}/merge",
                headers=headers,
                json=data
            )
            
            # Check the response
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"Failed to merge pull request: {response.status_code}",
                    "details": response.json()
                }
        
        except Exception as e:
            self.logger.error(f"Error merging pull request: {str(e)}")
            return {"error": str(e)}


# Utility functions

def is_git_repository(path: str) -> bool:
    """
    Check if a directory is a Git repository.
    
    Args:
        path: Path to the directory
        
    Returns:
        True if the directory is a Git repository, False otherwise
    """
    try:
        repo = Repo(path)
        return True
    except:
        return False


def create_git_repository(path: str, bare: bool = False) -> bool:
    """
    Create a new Git repository.
    
    Args:
        path: Path where to create the repository
        bare: Whether to create a bare repository
        
    Returns:
        True if successful, False otherwise
    """
    try:
        Repo.init(path, bare=bare)
        return True
    except Exception as e:
        logging.error(f"Error creating repository: {str(e)}")
        return False


def clone_git_repository(url: str, path: str) -> bool:
    """
    Clone a Git repository.
    
    Args:
        url: URL of the repository to clone
        path: Path where to clone the repository
        
    Returns:
        True if successful, False otherwise
    """
    try:
        Repo.clone_from(url, path)
        return True
    except Exception as e:
        logging.error(f"Error cloning repository: {str(e)}")
        return False

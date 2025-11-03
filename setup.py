import os

structure = {
    "": [
        "README.md",
        "requirements.txt",
        "config.yaml",
        ".gitignore",
        "run.py"
    ],
    "data": [],
    "tools": [
        "__init__.py",
        "csv_tools.py",
        "viz_tools.py",
    ],
    "agents": [
        "__init__.py",
        "assistant_agent.py",
        "prompts.py"
    ],
    "notebooks":[
        "experiment.ipynb"
    ],
    "tests":[
        "test_tools.py"
    ],

}

def create_structure(base_path):
    """Create all folders and files in the current directory."""
    for folder, files in structure.items():
        dir_path = os.path.join(base_path, folder)
        os.makedirs(dir_path, exist_ok=True)
        for file in files:
            file_path = os.path.join(dir_path, file)
            if not os.path.exists(file_path):
                with open(file_path, "w", encoding="utf-8") as f:
                    
                    if file.endswith(".py"):
                        f.write(f"# {file}\n# Auto-generated placeholder\n\n")
                    elif file.endswith(".md"):
                        f.write("# Intelligent CSV Analyst\n\n")
                    else:
                        f.write("")
    print("\nCompact project structure created successfully!")
    print(f"Location: {base_path}")
    print("\nFolders created:")
    for folder in structure.keys():
        print(" -", folder or ".")
   

if __name__ == "__main__":
    current_dir = os.getcwd()
    create_structure(current_dir)

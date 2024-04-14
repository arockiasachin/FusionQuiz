import subprocess

def create_requirements():
    # Using the full path to the conda executable
    full_path_to_conda = r'C:\Users\SachinSlade\anaconda3\Scripts\conda.exe'  # Adjust the path as necessary
    result = subprocess.run([full_path_to_conda, 'list', '--export'], capture_output=True, text=True)
    packages = result.stdout.splitlines()

    with open('requirements.txt', 'w') as f:
        for package in packages:
            parts = package.split('=')
            if len(parts) == 3:
                f.write(f"{parts[0]}=={parts[1]}\n")

if __name__ == "__main__":
    create_requirements()

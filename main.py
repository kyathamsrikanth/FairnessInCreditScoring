
import subprocess



if __name__ == '__main__':
    # List of Python files to run
    python_files = ['postprocess_reject_option_classification.py', 'preprocess_rw.py', 'postproces_equalised_odds.py','inprocess_meta_algorithm.py','inprocess_adversarial.py']

    # Loop through each file and run it
    for file in python_files:
        print(f"Running {file}...")
        result = subprocess.run(['python', file], capture_output=True, text=True)
        print(result.stdout)
        print(f"{file} finished.\n")

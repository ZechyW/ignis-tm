import subprocess
import os

my_env = os.environ.copy()

for i in range(2000, 3000):
    my_env["PYTHONHASHSEED"] = str(i)
    print(subprocess.check_output("python lda.py", env=my_env).decode().strip())

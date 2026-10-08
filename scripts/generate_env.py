import json
import os
import platform
import subprocess
import sys

import numpy as np
import optuna
import pandas as pd
import scipy
import sklearn
import statsmodels
import xgboost as xgb

def get_env_info():
    os_name = platform.platform()
    if "Windows-10" in os_name:
        try:
            build = int(platform.version().split(".")[2])
            if build >= 22000:
                os_name = os_name.replace("Windows-10", "Windows-11")
        except Exception:
            pass
    cpu_model = platform.processor()
    logical_cpus = os.cpu_count()

    # Query CPU info via powershell or wmic if on Windows
    cpu_name = cpu_model
    try:
        cmd = 'powershell -NoProfile -Command "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"'
        res = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        if res.returncode == 0 and res.stdout.strip():
            cpu_name = res.stdout.strip()
    except Exception:
        pass

    info = {
        "python_version": sys.version.split()[0],
        "platform": os_name,
        "cpu_model": cpu_name,
        "logical_cpus": logical_cpus,
        "nthread_training": 4,
        "nthread_latency": 1,
        "packages": {
            "xgboost": xgb.__version__,
            "scikit-learn": sklearn.__version__,
            "statsmodels": statsmodels.__version__,
            "scipy": scipy.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "optuna": optuna.__version__,
        }
    }
    return info

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    env_info = get_env_info()
    with open("results/env.json", "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)
    print("Saved results/env.json successfully:")
    print(json.dumps(env_info, indent=2))

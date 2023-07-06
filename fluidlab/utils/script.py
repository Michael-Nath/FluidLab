import numpy as np
rho_values = np.linspace(0, 50, 20)
lambda_values = np.linspace(240, 400, 10)
mu_values = np.linspace(-5, 300, 20)

mappings = {
    "rho": rho_values,
    "lambda": lambda_values,
    "mu": mu_values
}

with open("./exps/pour_params.txt", "a+") as f:
    for typ in mappings.keys():
        for b in mappings[typ]:
            for i in mappings[typ]:
                f.write(f"python3 fluidlab/run.py --cfg_file configs/exp_pouring.yaml --record --path {typ}-pour --phys {typ} --base {b} --inj {i}\n")

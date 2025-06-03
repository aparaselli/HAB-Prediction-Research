# HAB-Prediction-Research
Salinty.csv and Temperature.csv aquired from SIO https://shorestations.ucsd.edu/publications/data/
scoos_HAB_SIO.csv from https://habs.sccoos.org/scripps-pier


### Bloom Forecast Inference Guide

---

#### 1. Project Directory Structure

```
Code/
└─ Scripts/
   ├─ configs/
   │  └─ biolum_config.yaml   ← example configuration
   ├─ outputs/
   │  └─ bloom_forecast.json  ← generated after running inference
   └─ forecast.py             ← inference script
```

---

#### 2. Create a Configuration File

1. Navigate to `Code/Scripts/configs/`.
2. Copy the sample file or create a new one, e.g.:

```bash
cp biolum_config.yaml my_config.yaml
```

3. Edit the new file (`my_config.yaml`) and fill in **each variable**:

| Key | Description |
|-----|-------------|
| `data_path` | Path to the **“back-up” dataset** used for predictions |
| `parameters_path_1wk` | Path to the **1-week** model parameters |
| `parameters_path_2wk` | Path to the **2-week** model parameters |
| `parameters_path_3wk` | Path to the **3-week** model parameters |
| `target` | Name of the variable to forecast |
| `json_key_path` | Google Drive service-account **JSON key** file |
| `bloom_thresh` | Threshold: if a single model’s forecast ≥ this value, classify as “bloom” |
| `samp` | Use every *samp*-th model to reduce over-fitting |
| `n` | Total number of models in the ensemble |
| `p` | Percentage of models that must predict “bloom” for the final label **Likely** |

> **Tip:** Keep paths **relative** to `Code/Scripts/` when possible to simplify execution.

Example snippet:

```yaml
# my_config.yaml
data_path: ../data/bloom_backup.csv
parameters_path_1wk: ../models/1wk_params.pkl
parameters_path_2wk: ../models/2wk_params.pkl
parameters_path_3wk: ../models/3wk_params.pkl
target: Chlorophyll_a
json_key_path: ../secrets/my_service_account.json
bloom_thresh: 3000
samp: 5
n: 100
p: 60
```

---

#### 3. Obtain & Add a Google Service-Account Key

1. Visit **https://console.cloud.google.com**.
2. Create (or select) a project and enable the **Google Drive API**.
3. Navigate to **APIs & Services → Credentials → + Create Credential → Service account key**.
4. Choose **JSON** as the key type and download the file.
5. Save the file to a secure location (e.g., `Code/Scripts/secrets/`) and update `json_key_path` in your YAML.

---

#### 4. Run Inference

From `Code/Scripts/`:

```bash
python3 forecast.py configs/my_config.yaml
```

- Replace `my_config.yaml` with the name of your configuration file.
- The script will read the config, load models, and produce forecasts.

---

#### 5. Output

- A JSON file named **`bloom_forecast.json`** is written to:

```
Code/Scripts/outputs/bloom_forecast.json
```

This file contains the predicted bloom likelihoods and confidence intervals for the upcoming weeks.

---

**You’re all set!** Customize your YAML, run the command, and check the output JSON for your bloom forecasts.

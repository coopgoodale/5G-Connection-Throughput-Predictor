# 5G Connection Throughput Predictor

This course project predicts whether the **next recorded 5G measurement window**
has low, medium, or high throughput. It trains a random forest on the
Lumos5G-v1.0 CSV linked from the course slides, compares it with a
last-observed-band baseline, and creates a browser replay of held-out runs.
The dashboard is a **recorded replay**, not a live 5G measurement.

## Get started

You need Python 3.10 or newer. The reference repository supplies the CSV;
the CSV is not included in this project folder.

```sh
git clone --depth 1 https://github.com/av-troshin77/5gthroughput.git reference-5gthroughput
cd 5g-throughput-predictor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py --data ../reference-5gthroughput/Lumos5G-v1.0.csv
python -m http.server 8000 --directory site
```

Open http://localhost:8000. On Windows, activate with
**.venv\Scripts\activate** and use **python** instead of **python3**.
Run the clone command from the parent folder containing this project.
Alternatively, download the CSV from the
[Lumos5G team](https://networking.umn.edu/lumos5g) and pass its path to
**--data**. The included **site/results.json** lets you view the replay
before retraining; running the command replaces it with your results.

## What the code does

1. __train.py__ reads the 5G CSV, sorts by recording (__run_num__) and window
   (__seq_num__), and predicts the next row within the same recording.
   Features use the current and two previous throughput values plus current
   radio, motion, and network status measurements. GPS coordinates and the
   future row are excluded from the input.
2. Whole runs are split by ascending run number: approximately 70% training,
   15% validation, and 15% final test. The run numbers are used as a recording
   order; the dataset does not supply calendar timestamps for this split.
   Windows never cross a recording boundary.
3. The low/medium and medium/high cutoffs are the 33rd and 67th percentiles
   of **training targets only**. Missing radio values and the source's large
   invalid-value sentinel are handled inside the preprocessing pipeline.
4. The model uses a random forest. The baseline predicts that the next band
   equals the current band. Both are scored on the same validation and test
   windows with macro-F1, accuracy, per-class recall, and a confusion matrix.
5. **site/index.html**, **site/styles.css**, and **site/app.js** show the
   held-out predictions. Play, reveal, or seek through a run; the next measured
   band appears only after it is revealed. The dashboard also shows the full
   held-out test metrics and can jump to a mistaken transition.

The output file **site/results.json** contains model predictions, baseline
predictions, measured values, and metrics for the held-out runs. The browser
replays these exported predictions; it does not train a model or contact a
remote prediction service.

With the CSV linked above, the included result is **0.832 macro-F1** on 18
held-out test runs, compared with **0.813** for the last-band baseline.
Medium-band recall is **0.692**, lower than Low and High. These are measured
results from this split, not promised performance on another network.

## Evaluation and limitations

The split keeps whole runs separate, which avoids using adjacent windows from
the same run on both sides of the test boundary. A high test score would still
describe this dataset and collection setup, not every 5G network. The next
sample is the next recorded window in a run, not an independently timed live
forecast. Accuracy on band changes is reported separately because a
last-band predictor tends to fail there.

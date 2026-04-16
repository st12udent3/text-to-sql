# Text-to-SQL: LSTM Seq2Seq vs Fine-Tuned T5 on Spider

NLP group project comparing a from-scratch LSTM Seq2Seq model with a fine-tuned T5 transformer
for cross-database natural language to SQL translation on the Spider benchmark.

Research question: **Can a fine-tuned pretrained Transformer outperform a from-scratch LSTM
Seq2Seq model for cross-database natural language to SQL translation?**

## Project Structure

```
text-to-sql/
├── model.ipynb                 LSTM Seq2Seq: preprocessing, training, inference
├── t5_01_finetune.ipynb        T5-small fine-tuning (run on Colab)
├── t5_02_inference.ipynb       T5-small inference (run on Colab)
├── best_seq2seq_lstm.pt        Trained LSTM checkpoint (~50 MB)
├── predictions/
│   ├── gold.txt                Ground truth SQL + db_id for dev set
│   ├── pred_lstm.txt           LSTM predictions on dev set
│   └── pred_t5.txt             T5 predictions on dev set
├── eval_results/
│   ├── spider_lstm.txt         Spider official eval output (LSTM)
│   ├── ts_lstm.txt             Test-suite eval output (LSTM)
│   ├── spider_t5.txt           Spider official eval output (T5)
│   └── ts_t5.txt               Test-suite eval output (T5)
├── spider_data/spider_data/    Spider dataset (not in git, see below)
│   ├── train_spider.json
│   ├── train_others.json
│   ├── dev.json
│   ├── tables.json
│   └── database/               SQLite files per db_id
├── data/testsuitedatabases/    Test-suite augmented DBs (not in git)
├── requirements.txt
└── README.md
```

## Setup

### 1. Python environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
pip install -r requirements.txt
```

Python 3.13 was used during development.

### 2. Download Spider dataset

Download from https://yale-lily.github.io/spider and extract to `spider_data/spider_data/`
so that the paths `spider_data/spider_data/dev.json` and `spider_data/spider_data/database/`
exist.

### 3. Download test-suite databases

Needed for Test-Suite Accuracy evaluation.
Download from https://github.com/taoyds/test-suite-sql-eval (Google Drive link in README)
and extract to `data/testsuitedatabases/`. Expected structure:
`data/testsuitedatabases/database/<db_id>/<db_id>.sqlite`.

For Spider dev databases not covered by the test-suite dump, copy from the regular Spider set:

```powershell
$tsDir = "data\testsuitedatabases\database"
$spiderDir = "spider_data\spider_data\database"
Get-ChildItem $spiderDir -Directory | ForEach-Object {
    $target = Join-Path $tsDir $_.Name
    if (-not (Test-Path $target)) {
        Copy-Item $_.FullName $target -Recurse
    }
}
```

### 4. Clone evaluation scripts

The official Spider and Test-Suite evaluation scripts are used as-is. Clone them as siblings
to this repo:

```bash
cd ..
git clone https://github.com/taoyds/spider.git
git clone https://github.com/taoyds/test-suite-sql-eval.git
```

### 5. Download NLTK data

```bash
python -c "import nltk; [nltk.download(p) for p in ['punkt', 'punkt_tab', 'wordnet', 'omw-1.4']]"
```

## Running the Pipeline

### LSTM (local)

Open `model.ipynb` and run top-to-bottom. Sections:

1. **Data loading & exploration** (cells 1-30)
2. **Preprocessing**: schema linking, query normalization, input construction (cells 31-37)
3. **Data augmentation**: column/value replacement (cells 38-46)
4. **Tokenization & vocabulary** (cells 47-63)
5. **Model definition**: Encoder, Decoder, Additive Attention, Seq2Seq (cells 64-75)
6. **Training**: 15 epochs with teacher forcing, ReduceLROnPlateau scheduler (cells 76-91)
7. **Inference & prediction file writing** (final cells)

Outputs `predictions/pred_lstm.txt` and `predictions/gold.txt`.

### T5-small (Colab)

1. Upload `spider_data/` JSONs (train_spider.json, train_others.json, dev.json, tables.json)
   to your Google Drive under `MyDrive/text-to-sql/spider_data/`.
2. Run `t5_01_finetune.ipynb` on Colab with GPU runtime. Fine-tunes `t5-small` for 5 epochs,
   saves checkpoint to Drive at `MyDrive/text-to-sql/t5_small_spider/final/`.
3. Run `t5_02_inference.ipynb` on Colab with GPU runtime. Loads the checkpoint, runs inference
   on `dev.json`, downloads `pred_t5.txt` and `gold.txt` at the end.
4. Copy downloaded files into `predictions/`.

### Evaluation (local)

Run from `text-to-sql/` working directory.

**Spider official eval (Exact Set Match + Execution Accuracy):**

```powershell
python ..\spider\evaluation.py `
    --gold predictions\gold.txt `
    --pred predictions\pred_lstm.txt `
    --db spider_data\spider_data\database `
    --table spider_data\spider_data\tables.json `
    --etype all 2>&1 | Tee-Object eval_results\spider_lstm.txt
```

Replace `pred_lstm.txt` with `pred_t5.txt` and `spider_lstm.txt` with `spider_t5.txt` to
evaluate T5.

**Test-Suite eval:**

```powershell
python ..\test-suite-sql-eval\evaluation.py `
    --gold predictions\gold.txt `
    --pred predictions\pred_lstm.txt `
    --db data\testsuitedatabases\database `
    --table spider_data\spider_data\tables.json `
    --etype all `
    --plug_value *> eval_results\ts_lstm.txt
```

The `--plug_value` flag plugs gold values into predictions before execution, making the
evaluation robust to value prediction errors (LSTM in particular cannot predict literal
values reliably).

## Dataset

- **Spider** (Yu et al., 2018). Cross-database semantic parsing dataset with ~7,000 training
  and ~1,000 development examples across 200+ databases. Test set is hidden.
- **Train split**: `train_spider.json` + `train_others.json`
- **Dev split**: `dev.json` (used for final evaluation)
- **Databases unseen at test time**: the 20 dev databases do not overlap with the 140 train
  databases, enforcing true cross-database generalization.

## Models

### LSTM Seq2Seq

- Learned token embeddings (input and output vocabularies built from training data)
- Encoder: embedding + bidirectional LSTM
- Additive (Bahdanau) attention
- Decoder: embedding + LSTM + attention + output projection
- Beam search decoding (beam width 3) with UNK penalty and repetition penalty

### T5-small

- Pretrained `t5-small` (60M parameters) fine-tuned on Spider
- Input format: `translate to SQL: <question> | schema: <table1(col1,col2) | table2(...) ...>`
- Output format: SQL query as-is (no normalization)
- Fine-tuning: 5 epochs, batch size 16, learning rate 3e-4, fp16

## Evaluation Metrics

- **Exact Set Match (ESM)**: SQL clauses compared as unordered sets, values masked.
- **Execution Accuracy (EX)**: Predicted SQL executed on the target SQLite database and
  result set compared to gold. Unordered comparison except when ORDER BY is present.
- **Test-Suite Accuracy (TS)**: Predictions executed against multiple augmented database
  instances per schema; correct only if result matches on all of them. Stricter than EX.

All three metrics are additionally reported by **hardness level** (easy, medium, hard,
extra hard) based on Spider's official query complexity annotation.

## Requirements

See `requirements.txt`. Main dependencies:

- torch, transformers, datasets, accelerate
- sqlglot (SQL normalization)
- sqlparse, nltk (test-suite eval)
- pandas, matplotlib, seaborn

## References

- Yu et al. (2018). Spider: A large-scale human-labeled dataset for complex and cross-domain
  semantic parsing and text-to-SQL task. EMNLP.
- Raffel et al. (2020). Exploring the limits of transfer learning with a unified text-to-text
  transformer. JMLR.
- Zhong et al. (2020). Semantic Evaluation for Text-to-SQL with Distilled Test Suites. EMNLP.
- Spider evaluation scripts: https://github.com/taoyds/spider
- Test-suite evaluation scripts: https://github.com/taoyds/test-suite-sql-eval

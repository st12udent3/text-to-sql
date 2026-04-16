# Text-to-SQL: LSTM Seq2Seq vs Fine-Tuned T5 on Spider

Group project comparing a from-scratch LSTM Seq2Seq model with a fine-tuned T5-base
transformer on the Spider text-to-SQL benchmark.

Research question: Can a fine-tuned pretrained Transformer outperform a from-scratch LSTM
Seq2Seq model for cross-database natural language to SQL translation?

## Project Structure

```
text-to-sql/
├── model.ipynb                 LSTM Seq2Seq: preprocessing, training, inference
├── t5_01_finetune.ipynb        T5-base fine-tuning (Colab)
├── t5_02_inference.ipynb       T5-base inference (Colab)
├── best_seq2seq_lstm.pt        Trained LSTM checkpoint
├── predictions/
│   ├── gold.txt                Ground truth SQL + db_id for dev set
│   ├── pred_lstm.txt           LSTM predictions on dev set
│   └── pred_t5.txt             T5 predictions on dev set
├── eval_results/
│   ├── spider_lstm.txt         
│   └── spider_t5.txt           
├── spider_data/spider_data/    Spider dataset
├── requirements.txt
└── README.md			
```

The fine-tuned T5-base checkpoint is ~900 MB and not included in the repo.
Download link: https://drive.google.com/drive/folders/1fFM6Hx4RECUUeDc9xoRn3TsD0S2dpK1y?usp=sharing

## External Dependencies

The Spider evaluation script is used from the official repo. Clone it as a sibling:
```
git clone https://github.com/taoyds/spider.git
```

## Setup

### 1. Python environment

Python 3.13.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Spider dataset

Download from https://yale-lily.github.io/spider and extract to `spider_data/spider_data/`
so that `spider_data/spider_data/dev.json` and `spider_data/spider_data/database/` exist.

### 3. Spider evaluation script

Clone as a sibling to this repo:

```
cd ..
git clone https://github.com/taoyds/spider.git
```

### 4. NLTK data

```
python -c "import nltk; [nltk.download(p) for p in ['punkt', 'punkt_tab', 'wordnet', 'omw-1.4']]"
```

## Running the Pipeline

Run in this order:

### 1. LSTM (local)

Open `model.ipynb` and run top to bottom. Contains:

- Data loading and exploration
- Preprocessing: schema linking, query normalization, input construction
- Data augmentation
- Tokenization and vocabulary construction
- Model definition: Encoder, Bahdanau Attention, Decoder, Seq2Seq
- Training: 15 epochs with teacher forcing
- Inference: generates `predictions/pred_lstm.txt` and `predictions/gold.txt`

Outputs the trained checkpoint as `best_seq2seq_lstm.pt`.

### 2. T5-base (Colab)

Upload the Spider JSONs (`train_spider.json`, `train_others.json`, `dev.json`,
`tables.json`) to Google Drive under `MyDrive/text-to-sql/spider_data/`.

**Fine-tune**: Run `t5_01_finetune.ipynb` on Colab with GPU. 8 epochs, batch 16,
learning rate 1e-4. Checkpoint saved to Drive under
`MyDrive/text-to-sql/t5_base_spider/final/`.

**Inference**: Run `t5_02_inference.ipynb` on Colab with GPU. Loads the checkpoint,
generates SQL for all 1034 dev examples, downloads `pred_t5.txt` and `gold.txt`.

Copy the two downloaded files into `predictions/`.

### 3. Evaluation (local)

Run from the `text-to-sql/` working directory:
LSTM
```
python ..\spider\evaluation.py `
    --gold predictions\gold.txt `
    --pred predictions\pred_lstm.txt `
    --db spider_data\spider_data\database `
    --table spider_data\spider_data\tables.json `
    --etype all 2>&1 | Tee-Object eval_results\spider_lstm.txt
```
T5-base
```
python ..\spider\evaluation.py `
    --gold predictions\gold.txt `
    --pred predictions\pred_t5.txt `
    --db spider_data\spider_data\database `
    --table spider_data\spider_data\tables.json `
    --etype all 2>&1 | Tee-Object eval_results\spider_t5.txt
```

## Dataset

Spider (Yu et al., 2018). Cross-database semantic parsing dataset.

- Train split: `train_spider.json` + `train_others.json` (~8,600 examples)
- Dev split: `dev.json` (1,034 examples on 20 unseen databases)
- Test split: hidden, not used here
- Databases in dev do not overlap with databases in train, enforcing cross-database
  generalization.

## Models

### LSTM Seq2Seq (baseline)

...

### T5-base (fine-tuned)

- Pretrained `t5-base` (220M parameters)
- Input format: `translate English to SQL | <db_id> | <question> | schema: <table(cols) ... | foreign_keys: ...>`
- 8 epochs, batch 16, learning rate 1e-4, fp16
- Best checkpoint selected by validation loss

## Evaluation Metrics

- **Exact Set Match**: SQL clauses parsed and compared as unordered sets, literal values
  ignored.
- **Execution Accuracy**: predicted SQL executed against the target SQLite database,
  result set compared to gold.

Both metrics are reported on Spider's dev set and broken down by hardness level
(easy, medium, hard, extra).

## Results

| Metric | LSTM | T5-base |
|---|---|---|
| Exact Match (all) | 0.000 | 0.425 |
| Execution (all) | 0.001 | 0.432 |

## References

- Yu et al. (2018). Spider: A Large-Scale Human-Labeled Dataset for Complex and
  Cross-Domain Semantic Parsing and Text-to-SQL Task. EMNLP.
- Raffel et al. (2020). Exploring the Limits of Transfer Learning with a Unified
  Text-to-Text Transformer. JMLR.
- Spider evaluation scripts: https://github.com/taoyds/spider
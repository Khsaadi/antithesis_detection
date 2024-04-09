# Using Pre-Trained Language Models in an End-to-End Pipeline for Antithesis Detection
This repository contains the official implementation of "Using Pre-Trained Language Models in an End-to-End Pipeline for Antithesis Detection" accepted 
in LREC-2024 with the newly proposed antithesis dataset (**antithesis_dataset.csv**) publicly available

**Installation**

Run command below to install the environment (using python3.9):

```
pip install -r requirements.txt
```

**Fine-tuning**

Run command below to fine-tune encoder language model, e.g., BERT for the antithesis detection task:

```
python antithesis_detection.py 
```

**Results**

On test set of the antithesis dataset: 
<img width="1076" alt="anti_res" src="https://github.com/Khsaadi/antithesis_detection/assets/58224339/1aa6b208-f24d-449f-a693-2efafaa6fd44">


**Cite**

```
@article{saadikuh2024Anti,
  title={Using Pre-trained Language Models in an End-to-End Pipeline for Antithesis Detection},
  author={Kuhn, Ramona, and Saadi, Khouloud and Mitrovic, Jelena and Granitzer, Michael},
  journal={LREC},
  year={2024}
}
```

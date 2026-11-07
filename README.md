# edge-medvqa

Fine-tuning a **small vision-language model** to answer questions about radiology images, then compressing it to run on an ordinary **CPU**.

- **Model:** [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct), a SigLIP vision encoder plus the SmolLM2 decoder, about 0.26B parameters, Apache 2.0.
- **Data:** [VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad): CT, MRI and X-ray images with clinician-written questions. 1,793 train / 451 test pairs, CC0 licence.
- **Fine-tuning:** **LoRA** on the text decoder's attention layers only. Only 0.7% of the weights (1.8M of 258M) are trained and the vision encoder stays frozen.
- **Compression:** LoRA merged into the weights, then **dynamic int8 quantisation** of every linear layer, using `torchao` or the older built-in `torch.ao` API.
- **Evaluation:** closed (yes/no) accuracy, open exact match and open recall, plus model size and latency per answer, for each variant.

Real questions from the VQA-RAD test set:

```text
Q: is there evidence of an aortic aneurysm?                          answer: yes
Q: is there airspace consolidation on the left side?                 answer: yes
Q: is there any intraparenchymal abnormalities in the lung fields?   answer: no
```

## Why this setup

| Choice | Reason |
|---|---|
| 256M-parameter VLM | Fits a free Colab T4 for training and a laptop CPU for inference. One 512×512 image is only 64 visual tokens. |
| LoRA instead of full fine-tuning | Trains a few million parameters instead of 256M, so it needs less memory and less data, and the result is a few-MB adapter file. |
| Text decoder only | The vision encoder already sees images well; the model mainly has to learn the vocabulary and answer style of radiology. |
| Answer-only loss | The loss ignores the prompt and image tokens (labels = −100), so the model learns to answer, not to repeat the question. |
| Merge + int8 | After merging, inference costs the same as the base model. int8 weights are about 4× smaller than float32, with no calibration data needed. |

## Quick start

```bash
git clone https://github.com/zubairzulfiqar/edge-medvqa && cd edge-medvqa
pip install -e ".[dev]"

python -m medvqa.evaluate --limit 50                           # zero-shot baseline
python -m medvqa.train --epochs 2 --out runs/lora              # GPU recommended
python -m medvqa.evaluate --adapter runs/lora/adapter          # fine-tuned
python -m medvqa.evaluate --adapter runs/lora/adapter --int8 --device cpu   # edge version
```

The [Colab notebook](notebooks/edge_medvqa_colab.ipynb) runs all of it on a free T4 GPU and prints the comparison table.

## Results

**Full run (Colab T4, 451 test questions):** run the notebook and paste its table here.

**CI check (CPU, 8 training pairs, 4 LoRA steps, 3 test questions):** this only proves the whole pipeline runs on the real model. The scores are not meaningful.

| Variant | Closed acc. | Size (MB) | Latency / answer | Device |
|---|---|---|---|---|
| LoRA merged, float32 | 1/3 | 1026 | 1.50 s | GitHub runner CPU |
| LoRA merged, int8 (torchao) | 2/3 | 348 | 0.98 s | GitHub runner CPU |

- The size and latency are already real. int8 makes the model **2.9× smaller** and **about 1.5× faster** on the same CPU.
- The accuracy on 3 questions after 4 training steps means nothing; the full run gives the real numbers.
- LoRA trains 1,843,200 of 258,328,128 parameters, and the training loss fell from 5.4 to 2.6 within those 4 steps.

## How it works

```text
VQA-RAD sample ──► chat prompt: [image] + instruction + question
                                │
     training:  + assistant answer ──► labels = answer tokens only ──► LoRA update
     inference: greedy decoding, max 16 new tokens ──► first word/phrase ──► metrics
```

- `medvqa/data.py`: loads VQA-RAD, builds the chat messages and masks the labels.
- `medvqa/model.py`: loads the model (works across `transformers` versions), picks the LoRA targets, batches the data, and generates answers.
- `medvqa/train.py`: a plain PyTorch loop with AdamW, linear warm-up and decay, gradient clipping, and float16 gradient scaling on older GPUs.
- `medvqa/evaluate.py`: evaluates the zero-shot, fine-tuned and int8 variants and writes a markdown table.
- `medvqa/compress.py`: merges LoRA, applies dynamic int8 quantisation, and measures model size.
- `medvqa/metrics.py`: closed accuracy, open exact match and open recall, with answer normalisation.

**Tests:** 7 unit tests run anywhere. 3 quantisation tests check both int8 backends against the float32 output. 3 end-to-end tests download the real model and data and run on CPU in CI:
- the label masking keeps only the answer;
- LoRA targets only the text decoder;
- training, evaluation and the int8 model all run, and the int8 model is under 60% of the float32 size.

## Limitations

- VQA-RAD is small: 451 test questions, and many open answers appear only once. Differences of a few points are within noise.
- The exact-match metric is strict ("left lung" ≠ "lung"); open recall gives partial credit.
- Dynamic int8 speeds up CPUs with int8 instructions (AVX-512 VNNI, ARM dot-product). On others the gain is mostly in size.
- Next steps: ONNX Runtime or llama.cpp (GGUF) export, static quantisation with calibration, and a larger model (SmolVLM-2.2B) for accuracy.
- This is a research exercise, not a diagnostic tool.

## References

- J. J. Lau et al., "A dataset of clinically generated visual questions and answers about radiology images", *Scientific Data* 5, 2018.
- E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.
- A. Marafioti et al., "SmolVLM: Redefining small and efficient multimodal models", 2025.

## License

MIT (code). The model and dataset keep their own licences (Apache 2.0, CC0).

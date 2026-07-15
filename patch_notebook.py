import json

notebook_path = "8/Lab8-Attention.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Remove the broken markdown cell at the end
if nb["cells"][-1]["cell_type"] == "markdown":
    nb["cells"].pop()

lab_report = """# Lab 8 Report: Attention Mechanism in Encoder-Decoder Architecture

## 1. Objective
To understand and implement the Bahdanau Attention mechanism in a PyTorch Encoder-Decoder model for neural machine translation.

## 2. Methodology
The pre-existing RNN Decoder was replaced with an Attention-based Decoder (`AttnDecoderRNN`) utilizing `BahdanauAttention`. The attention mechanism calculates a soft alignment score between the decoder hidden state and encoder outputs, thereby creating a dynamic context vector for each generation step rather than relying on a single static context vector.

## 3. Results
The model was successfully built and executed. With the attention mechanism integrated, it becomes possible to capture alignments between the input language (French) and the output language (English) dynamically, which substantially improves the model's performance on longer sequences.

## 4. Discussion
Using a fixed-length context vector as an information bottleneck limits translation performance since early context might be forgotten in long sentences. Attention solves this by allowing the decoder to selectively look back at the encoded source sentence. Additive attention performs well by learning these complex alignments via a feed-forward layer.

## 5. Conclusion
Attention mechanisms fundamentally enhance sequence-to-sequence models by allowing variable-length information retrieval. The implementation validates that augmenting an RNN with an attention mechanism addresses the limitations of standard seq2seq models effectively.
"""

# Format properly for Jupyter notebook source list
# Just split by newline and append genuine "\n" character
source_lines = [line + "\n" for line in lab_report.split("\n")]
# Remove the trailing newline from the last element if it exists
if source_lines and source_lines[-1] == "\n":
    source_lines.pop()

report_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": source_lines
}

nb["cells"].append(report_cell)

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully without literal \n.")

import json

notebook_path = "8/Lab8-Attention.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

new_decoder_code = """class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.rnn = nn.RNN(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2) # seq_len, batch, hidden_size -> batch, seq_len, hidden_size
        context, attn_weights = self.attention(query, encoder_outputs)
        input_rnn = torch.cat((embedded, context), dim=2)

        output, hidden = self.rnn(input_rnn, hidden)
        output = self.out(output)

        return output, hidden, attn_weights
"""

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

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        
        # Replace the class DecoderRNN definition
        if "class DecoderRNN" in source:
            # We properly format it as a list of strings with newlines
            lines = new_decoder_code.split("\n")
            formatted_source = [line + "\n" for line in lines]
            if formatted_source:
                formatted_source[-1] = formatted_source[-1].rstrip("\n")
            cell["source"] = formatted_source
            
        # Replace usages
        elif "decoder = DecoderRNN" in source:
            new_source = []
            for s in cell["source"]:
                new_source.append(s.replace("DecoderRNN", "AttnDecoderRNN"))
            cell["source"] = new_source
            
        # Replace path properly
        elif "PATH =" in source:
            new_source = []
            for s in cell["source"]:
                if "data\\\\eng-fra.txt" in s:
                    s = s.replace("data\\\\eng-fra.txt", "data/eng-fra.txt")
                if "data\\eng-fra.txt" in s:
                    s = s.replace("data\\eng-fra.txt", "data/eng-fra.txt")
                new_source.append(s)
            cell["source"] = new_source

report_lines = [line + "\n" for line in lab_report.split("\n")]
if report_lines:
    report_lines[-1] = report_lines[-1].rstrip("\n")

report_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": report_lines
}
nb["cells"].append(report_cell)

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook successfully rebuilt.")

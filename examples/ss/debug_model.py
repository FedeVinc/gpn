import gpn.model
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
from pathlib import Path 
from gpn.model import ConvNetModel, ConvNetConfig


config_model = ConvNetConfig(
  dilation_cycle=6,
  dilation_double_every=1,
  dilation_max=32,
  hidden_size=512,
  initializer_range=0.02,
  kernel_size=9,
  model_type="ConvNet",
  n_layers=10,
  vocab_size=7
)

if __name__ == '__main__':
    model_path = "songlab/gpn-brassicales"
    seq = "CGGGTTAAAAATCTAGTTGTTATTATTAAAGGAAATAAAATATCCTCATAAAACAATTTGTTGTAATCTATCTTTGGGCTAATGTTCTTATCCTACAAGACGAACCCTGACCGTATTCGTCGTAGAAAAAAAATTGCTTCGATCCCATCATTGAGTTCAATAATCGGCGCACAAAGGCCGATTCATAAAAACTCTAGGCCCATTAAAGTAAAGCCCATTCTCAACCCTATCCAGTCTCCCTGTATATATATATTTACGACACCAACCCAGCGTTGATATTTAATTTTCTTCAGTCAGAGATTTCGAAACCCTAGTCGATTTCGAGATCCAACTAACTCTGCTCCTTATCTCAGGTAAAATTCTCGCTCGAGAACTCAATTGCTTATCCAAAGTTCCAACTGAAGATGCTTTCCTACTGAATCTTAGGTTAATGTTTTGGATTTGGAATCTTACCCGAAATTTCTCTGCAGCTTGTTGAATTTGCGAAGTATGGGAGACGCTAGAGACAACGAAGCCTACGAGGAGGAGCTCTTGGACTATGAAGAAGAAGACGAGAAGGTCCCAGATTCTGGAAACAAAGTTAACGGCGAAGCTGTGAAAAAGTGAGTTTTATGGTTTCCTCGATATGTTTCATGTATACTACTGTGTGTTTAAATTTGTCGATTCTTAGATTACTACTTGATAACAAGTAGCAGTATGT"
    saving_path = Path.cwd() / 'images' / 'debug'
    print(f"The length of the sequence is {len(seq)}")
    convnet = ConvNetModel(config=config_model)

    # Tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.get_vocab()
    # convert the input sequence into a token sequence
    input_ids = tokenizer(seq, return_tensors="pt", return_attention_mask=False, return_token_type_ids=False)["input_ids"]
    print(f"The input shape is: {input_ids.shape}")
    print(f"A first snippet of the input is: {input_ids[0, :10]}")

    # model in eval mode
    print(f"Start Codon: {seq[489:492]}")  # Start codon
    pos = 489  # Let's mask the A and check the model predictions
    input_ids[0, pos] = tokenizer.mask_token_id
    print(f"New input: {input_ids}")

    convnet.eval()
    with torch.no_grad():
      output = convnet(input_ids=input_ids) #.logits
    
    print(f"logits shape: {output.last_hidden_layer.shape}")

"""     nucleotides = list('acgt')
    logits = output_logits[0, pos, [tokenizer.get_vocab()[nc] for nc in nucleotides]]
    probs = torch.nn.functional.softmax(logits, dim=0).numpy()
    probs_df = pd.DataFrame(dict(nucleotide=nucleotides, probability=probs))
    plt.figure(figsize=(10, 6)) 
    sns.barplot(data=probs_df, x="nucleotide", y="probability")
    # Saving the barplot
    plt.savefig(saving_path / 'barplot.png', dpi=300)
    plt.show()
 """




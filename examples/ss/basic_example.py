import gpn.model
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
from pathlib import Path 


def example_ss(model_path: str, seq: str, saving_path: Path):
    """
    Function that replicates the example from the notebook basic_example.ipynb

    """
    # get the tokenizer for the specified model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.get_vocab()
    # convert the input sequence into a token sequence
    input_ids = tokenizer(seq, return_tensors="pt", return_attention_mask=False, return_token_type_ids=False)["input_ids"]
    print(f"The input shape is: {input_ids.shape}")
    print(f"A first snippet of the input is: {input_ids[0, :10]}")
    # Get the pretrained model and set it to eval mode -> no grad required
    # Then calculate the embedding for the input sequence defined above
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    with torch.no_grad():
        embedding = model(input_ids=input_ids).last_hidden_state
    print(f"The embedding shape is: {embedding.shape}")
    # Let's do a quick visualization. We'll standardize the embeddings.
    embedding_df = pd.DataFrame(StandardScaler().fit_transform(embedding[0].numpy()))
    embedding_df.index.name = "Position"
    embedding_df.columns.name = "Embedding dimension"
    print(f"Dataframe embedding:\n{embedding_df}")
    # If you squint you can see the exon locations :)
    plt.figure(figsize=(10, 6))
    sns.heatmap(embedding_df.T, center=0, vmin=-3, vmax=3, cmap="coolwarm", square=True, xticklabels=100, yticklabels=100)
    plt.savefig(saving_path / 'heatmap.png', dpi=300)
    plt.show()
    ## Masked language modeling
    model_for_mlm = AutoModelForMaskedLM.from_pretrained(model_path)
    model_for_mlm.eval()
    print(f"Start Codon: {seq[489:492]}")  # Start codon
    pos = 489  # Let's mask the A and check the model predictions
    input_ids[0, pos] = tokenizer.mask_token_id
    print(f"New input: {input_ids}")

    with torch.no_grad():
        all_logits = model_for_mlm(input_ids=input_ids).logits
    
    print(f"logits shape: {all_logits.shape}")

    nucleotides = list('acgt')
    logits = all_logits[0, pos, [tokenizer.get_vocab()[nc] for nc in nucleotides]]
    probs = torch.nn.functional.softmax(logits, dim=0).numpy()
    probs_df = pd.DataFrame(dict(nucleotide=nucleotides, probability=probs))
    plt.figure(figsize=(10, 6)) 
    sns.barplot(data=probs_df, x="nucleotide", y="probability")
    # Saving the barplot
    plt.savefig(saving_path / 'barplot.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    model_path = "songlab/gpn-brassicales"
    seq = "CGGGTTAAAAATCTAGTTGTTATTATTAAAGGAAATAAAATATCCTCATAAAACAATTTGTTGTAATCTATCTTTGGGCTAATGTTCTTATCCTACAAGACGAACCCTGACCGTATTCGTCGTAGAAAAAAAATTGCTTCGATCCCATCATTGAGTTCAATAATCGGCGCACAAAGGCCGATTCATAAAAACTCTAGGCCCATTAAAGTAAAGCCCATTCTCAACCCTATCCAGTCTCCCTGTATATATATATTTACGACACCAACCCAGCGTTGATATTTAATTTTCTTCAGTCAGAGATTTCGAAACCCTAGTCGATTTCGAGATCCAACTAACTCTGCTCCTTATCTCAGGTAAAATTCTCGCTCGAGAACTCAATTGCTTATCCAAAGTTCCAACTGAAGATGCTTTCCTACTGAATCTTAGGTTAATGTTTTGGATTTGGAATCTTACCCGAAATTTCTCTGCAGCTTGTTGAATTTGCGAAGTATGGGAGACGCTAGAGACAACGAAGCCTACGAGGAGGAGCTCTTGGACTATGAAGAAGAAGACGAGAAGGTCCCAGATTCTGGAAACAAAGTTAACGGCGAAGCTGTGAAAAAGTGAGTTTTATGGTTTCCTCGATATGTTTCATGTATACTACTGTGTGTTTAAATTTGTCGATTCTTAGATTACTACTTGATAACAAGTAGCAGTATGT"
    saving_path = Path.cwd() / 'images' / 'debug'
    print(f"The length of the sequence is {len(seq)}")
    example_ss(model_path=model_path, seq=seq, saving_path=saving_path)







# genslm_embed

Generating [GenSLM](https://github.com/ramanathanlab/genslm) ORF and genome embeddings for viral genes and genomes.

## Disclaimer

**This is a wrapper repository for GenSLM. For any problems specific to GenSLM, please contact the above repository.** We did not develop GenSLM, so we make no guarantees about it.

## Installation

### Without GPUs

```bash
# setup torch first -- conda does this so much better than pip
mamba create -n genslm-embed -c pytorch -c conda-forge 'pytorch>=2.0' cpuonly python=3.10 pytorch-lightning=1.6.5

mamba activate genslm-embed

# install latest updates from this repository
pip install git+https://github.com/cody-mar10/genslm_embed.git
```

### With GPUs

```bash
# setup torch first -- conda does this so much better than pip
mamba create -n genslm-embed -c pytorch -c nvidia -c conda-forge 'pytorch>=2.0' pytorch-cuda=11.8 python=3.10 pytorch-lightning=1.6.5

mamba activate genslm-embed

# install latest updates from this repository
pip install git+https://github.com/cody-mar10/genslm_embed.git
```

This will install the executable `genslm`.

## Usage

You need a gene (ORF) FASTA file in nucleotide format. The genes should be in the order that they appear in each genome.

### 1. Tokenize

You first need to tokenize your genes:

```bash
genslm tokenizer -f FASTAFILE -o TOKENS.h5
```

See the help page (`genslm tokenizer -h`) for more explained options.

The output `TOKENS.h5` file has 2 fields:

- `tokens`, which are the integer tokens for each tokenized chunk of each gene
- `attn_mask`, which is used to mask out padded values

### 2. Embed tokenized sequences

Then you can use the corresponding GenSLM to embed your tokenized sequences:

```bash
genslm embed -t TOKENS.h5 -o OUTPUT.h5
```

See the help page (`genslm embed -h`) for more explained options, such as changing your compute device, model size, and the model caching directory.

Note: Be sure that you use the same model for tokenizing and embedding (`-i/--model-id`).

The embedding file `OUTPUT.h5` contains 1 field (`data`), which are the embeddings for each gene ORF.

### Genome embeddings

To get genome embeddings, you can average these ORF embeddings over each genome. We have not provided code in this repository, but we have an example for how to do this associated with our [PST manuscript](https://github.com/AnantharamanLab/protein_set_transformer/blob/main/manuscript/genome_embeddings/genome_average.ipynb). You can use the `pst graphify` utility to convert ORF embeddings to a graph format that keeps track of which genes belong to which genome to simplify genome averaging. The usage is described in the [PST repository](https://github.com/AnantharamanLab/protein_set_transformer).

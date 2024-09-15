# [Is a Good Description Worth a Thousand Pictures? Reducing Multimodal Alignment to Text-Based, Unimodal Alignment](https://openreview.net/forum?id=o1KMaRvFqB)

In this paper, we investigated whether the multimodal alignment problem could be effectively reduced to the unimodal alignment problem, wherein a language model would make a moral judgment purely based on a description of an image. Focusing on GPT-4 and LLaVA as two prominent examples of multimodal systems, we demonstrated, rather surprisingly, that this reduction can be achieved with a relatively small loss in moral judgment performance in the case of LLaVa, and virtually no loss in the case of GPT-4.

## First Time Setup:
This setup is specifically for the Mila cluster. Revision will be needed for other infrastructures.

### Prerequisites:
- Ensure you have access to the cluster.
- Ensure you have the necessary permissions to run interactive jobs.

### Setup Steps:
1. **Get an interactive job**:
   - Follow the cluster's documentation to start an interactive job.

2. **Clone the repository**:
   ```zsh
   cd ~
   git clone https://github.com/amimem/alignment.git
   ```

3. **Set permissions and run setup script**:
   ```zsh
   cd alignment
   chmod +x scripts/*.sh
   source scripts/setup.sh
   ```

### Future Use:
For future interactive jobs, simply run:
```zsh
source scripts/int.sh
```

### Running the sbatch Script:
To submit a batch job, run:
```zsh
source scripts/slurm.sh
```
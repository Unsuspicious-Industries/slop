import os
import sys
import argparse

# Add project root to path
sys.path.append(os.getcwd())

from src.analysis.experiment import ExperimentRunner

def main():
    parser = argparse.ArgumentParser(description="Run SLOP Experiment")
    parser.add_argument("--prompts", nargs="+", default=["a photo of a person"], help="List of prompts to generate")
    parser.add_argument("--prompt-file", default=None, help="Optional prompt file with 'group\tprompt' per line")
    parser.add_argument("--diffuser", default="hf-internal-testing/tiny-stable-diffusion-torch", help="Diffusion model ID")
    parser.add_argument("--encoder", default="openai/clip-vit-base-patch32", help="Encoder model ID")
    parser.add_argument("--output", default="experiments/test_run", help="Output directory")
    parser.add_argument("--batch-name", default="cli_run", help="Batch name used for output files")
    parser.add_argument("--steps", type=int, default=5, help="Number of diffusion steps")
    parser.add_argument("--device", default="cpu", help="Device to run on")
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(
        diffusion_model_id=args.diffuser,
        encoder_model_id=args.encoder,
        output_dir=args.output,
        device=args.device
    )

    prompt_groups = None
    prompts = args.prompts
    if args.prompt_file:
        prompt_groups = {}
        prompts = []
        with open(args.prompt_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "\t" in line:
                    group, prompt = line.split("\t", 1)
                else:
                    group, prompt = "ungrouped", line
                prompts.append(prompt)
                prompt_groups[prompt] = group

    runner.run_batch(prompts=prompts, num_steps=args.steps, batch_name=args.batch_name, prompt_groups=prompt_groups)
    runner.aggregate_data(batch_name=args.batch_name)

if __name__ == "__main__":
    main()

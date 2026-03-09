{
  description = "slop — diffusion trajectory & bias analysis";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true; # torch, cuda bits
        };

        py = pkgs.python311;

        # All analysis + client deps available locally.
        # Inference itself runs remotely (vast.ai); torch here is CPU-only for
        # testing hooks & trajectory code without a live server.
        pythonEnv = py.withPackages (ps: with ps; [
          # numerics
          numpy
          scipy
          scikit-learn
          umap-learn

          # viz
          matplotlib
          seaborn
          plotly
          streamlit

          # ML
          torch        # CPU-only; GPU runs in the remote Docker image
          torchvision
          transformers
          diffusers
          accelerate
          einops
          safetensors
          sentencepiece
          protobuf

          # vision utilities
          opencv4
          pillow

          # project utilities
          jsonlines
          tqdm
          pyyaml

          # dev / notebooks
          pytest
          ipython
          jupyter

          # type checking
          mypy
        ]);

      in {
        devShells.default = pkgs.mkShell {
          packages = [
            pythonEnv
            pkgs.git
            pkgs.openssh   # for SSH transport to vast.ai nodes
            pkgs.ruff      # linter/formatter
          ];

          shellHook = ''
            # Project root on PYTHONPATH so `from client.`, `from server.`,
            # `from shared.`, `from src.` all resolve without installs.
            export PYTHONPATH="${toString self}:''${PYTHONPATH:-}"

            echo "slop dev shell (python $(python --version | cut -d' ' -f2))"
          '';
        };
      });
}

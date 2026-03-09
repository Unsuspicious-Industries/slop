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
          overlays = [
            (final: prev: {
              pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
                (pyFinal: pyPrev: {
                  pynndescent = pyPrev.pynndescent.overridePythonAttrs (old: {
                    doCheck = false;
                  });
                })
              ];
            })
          ];
        };

        py = pkgs.python312;

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

          # ML
          torch        # CPU-only; GPU runs in the remote Docker image
          torchvision
          transformers
          diffusers
          accelerate
          peft
          einops
          safetensors
          sentencepiece
          protobuf
          openai
          aiohttp

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
            # Native libs required by pip-installed wheels (numpy, scipy, etc.)
            pkgs.zlib
            pkgs.stdenv.cc.cc.lib
            pkgs.libGL
            pkgs.glib
          ];

          shellHook = ''
            # Project root on PYTHONPATH so `from client.`, `from server.`,
            # `from shared.`, `from src.` all resolve without installs.
            export PYTHONPATH="${toString self}:''${PYTHONPATH:-}"

            # Expose Nix-provided native libs so pip wheels can find them.
            # (pip wheels link against system libz, libstdc++, libGL etc.)
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
              pkgs.zlib
              pkgs.stdenv.cc.cc.lib
              pkgs.libGL
              pkgs.glib
            ]}:''${LD_LIBRARY_PATH:-}"

            # Setup .venv if it doesn't exist
            if [ ! -d .venv ]; then
              echo "Creating .venv..."
              python -m venv .venv
            fi
            source .venv/bin/activate

            # Load .env if it exists
            if [ -f .env ]; then
              echo "Loading .env..."
              # Export each line of .env, stripping quotes if present
              export $(grep -v '^#' .env | sed "s/['\"]//g" | xargs)
              
              # Ensure VAST_API_KEY is set if VASTAI_API_KEY is present
              if [ -n "$VASTAI_API_KEY" ] && [ -z "$VAST_API_KEY" ]; then
                export VAST_API_KEY="$VASTAI_API_KEY"
              fi
            fi

            # Install requirements and vastai
            if [ -f requirements.txt ]; then
              echo "Checking/Installing requirements into .venv..."
              pip install --quiet -r requirements.txt
            fi

            # Packages not available from nixpkgs Python set
            # Keep these minimal so most of the environment comes from the flake.
            if ! python -c "import xai_sdk" >/dev/null 2>&1; then
              echo "Installing xai-sdk into .venv..."
              pip install --quiet xai-sdk
            fi

            if ! python -c "import torch_fidelity" >/dev/null 2>&1; then
              echo "Installing torch-fidelity into .venv..."
              pip install --quiet torch-fidelity
            fi

            if ! command -v vastai &> /dev/null; then
              echo "Installing vastai CLI..."
              pip install --quiet vastai
            fi

            echo "slop dev shell (python $(python --version | cut -d' ' -f2))"
            echo "VIRTUAL_ENV: $VIRTUAL_ENV"
          '';
        };
      }) // {
        packages.x86_64-linux.default = self.devShells.x86_64-linux.default.inputDerivation;
      };
}

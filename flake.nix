{
  description = "Kuchikae — prompt-conditioned voice transformation prototype";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs { inherit system; };

      kuchikae-tools = pkgs.symlinkJoin {
        name = "kuchikae-tools";
        paths = with pkgs; [
          python311
          uv
          ollama
          ffmpeg
          sox
          libsndfile
          portaudio
          git
        ];
        buildInputs = [ pkgs.makeWrapper ];
      };

    in rec {
      packages.default = kuchikae-tools;
      formatter = pkgs.alejandra;
      checks.format = pkgs.runCommand "kuchikae-format-check" {
        nativeBuildInputs = [ pkgs.alejandra ];
      } ''
        alejandra --check ${./.}
        touch $out
      '';

      devShells.default = pkgs.mkShell {
        packages = with pkgs; [
          python311
          uv
          ollama
          ffmpeg
          sox
          libsndfile
          portaudio
          git
        ];
        shellHook = ''
          export UV_PROJECT_ENVIRONMENT=.venv
          export PYTHONPATH=$PWD

          echo "========================"
          echo "  Kuchikae dev shell"
          echo ""
          echo "  Run: uv sync"
          echo "  Run: uv run pytest"
          echo "  Run: uv run kuchikae serve"
          echo "========================"
        '';
      };
    });
}

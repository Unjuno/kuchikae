{
  description = "Kuchikae — prompt-conditioned voice transformation prototype";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs { inherit system; };
    in rec {
      packages.default = pkgs.symlinkJoin {
        name = "kuchikae-tools";
        paths = with pkgs; [
          python311
          uv
          ffmpeg
          sox
          libsndfile
          portaudio
          git
        ];
        buildInputs = [ pkgs.makeWrapper ];
      };

      devShells.default = pkgs.mkShell {
        packages = with pkgs; [
          python311
          uv
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
          echo "  Run: uv run python app.py"
          echo "========================"
        '';
      };
    });
}

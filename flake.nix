{
  description = "Flake using pyproject.toml metadata";

  inputs = {
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { nixpkgs, pyproject-nix, ... }:
    let
      inherit (nixpkgs) lib;
      forAllSystems =
        function: lib.genAttrs lib.systems.flakeExposed (system: function nixpkgs.legacyPackages.${system});

      project = pyproject-nix.lib.project.loadPyproject {
        projectRoot = ./.;
      };
    in
    {
      packages = forAllSystems (pkgs: {
        default = pkgs.python3.pkgs.buildPythonPackage (
          project.renderers.buildPythonPackage { python = pkgs.python3; }
        );
      });

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = [
            (pkgs.python3.withPackages (project.renderers.withPackages { python = pkgs.python3; }))
          ];
        };
      });
    };
}

{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: let
  pkgs-unstable = import inputs.nixpkgs-unstable {system = pkgs.stdenv.system;};
  uv_bin = "${pkgs-unstable.uv}/bin/uv";
  uv_run = "${uv_bin} run";
  alej_run = "${pkgs.alejandra}/bin/alejandra";
in {
  packages = [
    pkgs.git
    pkgs.alejandra
  ];
  cachix.enable = false;

  env.UV_PROJECT_ENVIRONMENT = lib.mkForce "${config.devenv.root}/.venv";
  enterShell = ''
    if [ -f "$DEVENV_ROOT/.venv/bin/activate" ]; then
      source "$DEVENV_ROOT/.venv/bin/activate"
    fi
  '';

  languages = {
    nix.enable = true;
    python = {
      enable = true;
      version = "3.13";
      uv = {
        enable = true;
        package = pkgs-unstable.uv;
      };
    };
  };

  scripts = {
    fix_linting = {
      exec = ''
        ${uv_run} ruff format
        ${uv_run} ruff check --fix
        ${alej_run} --exclude ./.devenv --exclude ./.devenv.flake.nix .
        ${uv_run} tombi format
      '';
    };
    run_tests_all = {
      exec = ''
        devenv tasks run test:all
      '';
    };
    run_tests_local = {
      exec = ''
        devenv tasks run check:local
      '';
    };
  };

  tasks = {
    "test:fast" = {
      exec = ''
        ${uv_run} pytest -m 'not (simulation or hardware)' --reruns 3
      '';
    };
    "test:hardware" = {
      exec = ''
        ${uv_run} pytest -m 'hardware' --reruns 1
      '';
    };
    "test:all" = {
      exec = ''
        ${uv_run} pytest --reruns 3
      '';
    };
    "check:toml-lint" = {
      exec = ''
        ${uv_run} tombi check .
      '';
    };
    "check:python-lint" = {
      exec = ''
        ${uv_run} ruff check .
      '';
    };
    "check:python-types" = {
      exec = ''
        ${uv_run} ty check .
      '';
    };
    "check:local" = {
      after = [
        "test:fast"
        "check:python-lint"
        "check:python-types"
        "check:toml-lint"
      ];
    };
  };
}

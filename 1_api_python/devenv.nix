{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: let
  pkgs-rosetta = pkgs.pkgsx86_64Darwin;
  pkgs-unstable = import inputs.nixpkgs-unstable {system = pkgs.stdenv.system;};
  uv_bin = "${pkgs-unstable.uv}/bin/uv";
  uv_run = "${uv_bin} run --active";
  alej_run = "${pkgs.alejandra}/bin/alejandra";
  tombi_run = "${pkgs.tombi}/bin/tombi";

  modulesDir = ./.;
  nixFiles =
    if builtins.pathExists modulesDir
    then
      builtins.attrNames (
        lib.filterAttrs
        (name: type: type == "regular" && lib.hasSuffix ".nix" name && name != "devenv.nix")
        (builtins.readDir modulesDir)
      )
    else lib.warn "Folder ${toString modulesDir} not found – no module loaded" [];

  moduleImports = map (f: modulesDir + "/${f}") nixFiles;
in {
  imports = moduleImports;

  packages = [
    pkgs.git
    pkgs.tombi
    pkgs.ruff
    pkgs.alejandra
    pkgs.wget
    pkgs.gzip
  ];
  cachix.enable = false;
  languages = {
    c = {
      enable = false;
    };
    nix = {
      enable = true;
    };
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
        ${tombi_run} format
      '';
    };
    run_tests_all = {
      exec = ''
        devenv tasks run test:changes
      '';
    };
    run_tests_local = {
      exec = ''
        devenv tasks run check:local
      '';
    };
  };

  tasks = let
    uv_run = "${pkgs-unstable.uv}/bin/uv run --active";
    uv_build = "${pkgs-unstable.uv}/bin/uv build";
  in {
    "project:sync" = {
      exec = ''
        ${uv_run} sync
      '';
    };
    "package:build" = {
      exec = ''
        ${uv_bin} build
      '';
    };
    "docs:check" = {
      exec = ''
        export LC_ALL=C  # necessary to run in github action
        ${uv_run} sphinx-build -b singlehtml docs build/docs
      '';
      after = ["docs:clean"];
    };
    "docs:build" = {
      exec = ''
        export LC_ALL=C  # necessary to run in github action
        ${uv_run} sphinx-build -j auto -b html docs build/docs
        touch build/docs/.nojekyll  # prevent github from trying to build the docs
      '';
      after = ["docs:clean"];
    };
    "docs:clean" = {
      exec = ''
        rm -rf build/docs/*
      '';
      before = ["docs:build"];
    };
    "test:init" = {
      exec = ''
        rm -rf .testmondata*
        ${uv_run} pytest --testmon -m 'not (simulation or slow or plot)' --reruns 3
      '';
    };
    "test:changes" = {
      exec = ''
        ${uv_run} pytest --testmon --reruns 3
      '';
    };
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
    "test:simulation" = {
      exec = ''
        ${uv_run} pytest -m 'simulation' --reruns 1
      '';
    };
    "test:all" = {
      exec = ''
        ${uv_run} pytest --reruns 3
      '';
    };
    "test:coverage" = {
      exec = ''
        ${uv_run} coverage run -m pytest -m 'not hardware' --reruns 3
      '';
    };
    "check:coverage-report" = {
      exec = ''
        ${uv_run} coverage report -m
        ${uv_run} coverage xml
        ${uv_run} coverage html
      '';
      after = ["test:coverage"];
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
    "check:dependencies" = {
      exec = ''
        ${uv_run} pip-audit
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

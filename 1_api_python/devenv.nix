{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: let
    pkgs-unstable = import inputs.nixpkgs-unstable {system = pkgs.stdenv.system;};

    uv_base = "${pkgs-unstable.uv}/bin/uv";
    uv_run = "${uv_base} run --active";
    alej_run = "${pkgs.alejandra}/bin/alejandra";
    tombi_run = "${pkgs.tombi}/bin/tombi";
in {
  packages = [
    pkgs.git
    pkgs.tombi
    pkgs.ruff
    pkgs.alejandra # nix formatter
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
    fix_linting = {
      exec = ''
        ${uv_run} ruff format
        ${uv_run} ruff check --fix
        ${alej_run} --exclude ./.devenv --exclude ./.devenv.flake.nix .
        ${tombi_run} format
      '';
    };
  };

  tasks = {
    "project:sync" = {
      exec = ''
        ${uv_base} sync
      '';
    };
    "package:build" = {
      exec = ''
        ${uv_base} build
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
        ${uv_run} pytest --testmon -m 'not (hardware or unstable)' --reruns 3
      '';
    };
    "test:changes" = {
      exec = ''
        ${uv_run} pytest --testmon -m 'not (hardware or unstable)' --reruns 3
      '';
    };
    "test:fast" = {
      exec = ''
        ${uv_run} pytest -m 'not (hardware or unstable)' --reruns 3
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
    "test:coverage" = {
      exec = ''
        ${uv_run} coverage run -m pytest -m 'not hardware' --reruns 3
      '';
    };
    "check:coverage-report" = {
      exec = ''
        ${uv_run} coverage report -m
        ${uv_run} coverage xml
      '';
      after = ["test:coverage"];
    };
    "check:toml-lint" = {
      exec = ''
        ${tombi_run} check .
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

  enterShell = ''
    echo ""
  '';
}

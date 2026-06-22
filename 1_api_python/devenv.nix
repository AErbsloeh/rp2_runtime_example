{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: let
  pkgs-unstable = import inputs.nixpkgs-unstable {system = pkgs.stdenv.system;};
in {
  packages = [
    pkgs.git
    pkgs.tombi
    pkgs.ruff
    pkgs.iverilog
    pkgs.zlib # needed as dependency cocotb/ghdl under circumstances
    pkgs.alejandra # nix formatter
    pkgs.wget
    pkgs.gzip
    pkgs.gnutar
    pkgs.picotool
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

  scripts = let
    uv_run = "${pkgs-unstable.uv}/bin/uv run --active";
    alej_run = "${pkgs.alejandra}/bin/alejandra";
    tombi_run = "${pkgs.tombi}/bin/tombi";
  in {
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
    flash_dirty_jtag = {
      exec = ''
        if [[ -e $DEVENV_ROOT/external/dirtyJtag.uf2 ]]; then
          RELEASE=$(wget -qO- "https://api.github.com/repos/phdussud/pico-dirtyJtag/releases/latest" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')
          URL="https://github.com/phdussud/pico-dirtyJtag/releases/download/$RELEASE/dirtyJtag.uf2"
          echo "Downloading DirtyJTAG ($URL) ... "
          cd $DEVENV_ROOT/external && wget -q "$URL"
          echo "Done"
        else
          echo "RP2040 Executable found."
        fi

        picotool load --verify --force --execute dirtyJtag.uf2
      '';
      package = pkgs.bash;
    };
    get_latest_oss_cad_suite_release = {
      exec = ''
        echo $(wget -qO- "https://api.github.com/repos/YosysHQ/oss-cad-suite-build/releases/latest" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')
      '';
      package = pkgs.bash;
    };
    check_oss_cad_available = {
      exec = ''
        if [[ -e $DEVENV_ROOT/oss-cad-suite/VERSION ]]; then
            LATEST=$(get_latest_oss_cad_suite_release | tr -d "-")
            CURRENT=$(cat $DEVENV_ROOT/oss-cad-suite/VERSION)
            if [[ "$LATEST" != "$CURRENT" ]]; then
                echo "You have currently Version $CURRENT of the oss-cad-suite installed. A newer version ($LATEST) is available. If you want to update, please enter 'download_build_tools' into terminal."
            fi
        else
            echo "You don't have the oss-cad-suite installed. Please enter 'download_build_tools' into terminal"
        fi
      '';
      package = pkgs.bash;
    };
    download_build_tools = {
      exec = ''
        if [[ "$(uname -m)" == "arm64" ]]; then
          ARCH='arm64'
        elif [[ "$(uname -m)" == "x86_64" ]]; then
          ARCH="x64"
        else
          echo "Architecture not supported!"
          exit 1
        fi
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
           OS="linux"
        elif [[ "$OSTYPE" == "darwin"* ]]; then
          OS="darwin"
        else
          echo "OS not supported!"
          exit 1
        fi

        RELEASE=$(get_latest_oss_cad_suite_release)
        URL="https://github.com/YosysHQ/oss-cad-suite-build/releases/download/$RELEASE/oss-cad-suite-$OS-$ARCH-$(echo "$RELEASE" | tr -d "-").tgz"

        echo "Downloading OSS-CAD-Suite ($URL) ... "
        cd $DEVENV_ROOT && wget -qO- "$URL" | gunzip | tar xf -
        echo "Done"

        if [[ "$OS" == "darwin" ]]; then
          $DEVENV_ROOT/oss-cad-suite/activate
        fi
      '';
      package = pkgs.bash;
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
        ${uv_build}
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

  enterShell = ''
    #check_oss_cad_available
    #if [[ -e $DEVENV_ROOT/oss-cad-suite/environment ]]; then
    #  echo "Source oss-cad-suite from local project"
    #  cd $DEVENV_ROOT && source /oss-cad-suite/environment
    #elif [[ -e $HOME/oss-cad-suite/environment ]]; then
    #  echo "Source oss-cad-suite from home directory"
    #  cd $HOME && source oss-cad-suite/environment
    #else
    #  echo "oss-cad-suite not found"
    #fi
    #cd $DEVENV_ROOT
  '';
}

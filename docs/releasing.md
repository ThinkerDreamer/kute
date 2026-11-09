# Releasing

kute follows [Semantic Versioning](https://semver.org/). Before 1.0 the minor version carries
both new features and breaking changes (a new config key, a changed `meta.toml` field, a renamed
command), and the patch version carries fixes only. The version lives in one place,
`Cargo.toml`; `nix/package.nix` reads it from there, `kute --version` and each session's
`meta.toml` (`kute_version`) report it, and the release tarball is named after it.

`CHANGELOG.md` is generated from the Conventional Commits on `main` by
[git-cliff](https://git-cliff.org) with `cliff.toml`. `feat`, `fix`, `perf` and `build` commits
show up in it; `docs`, `test`, `ci`, `refactor` and `chore` commits don't (the user guides from
#1 are the exception, from before PRs were rebased). Pull requests are rebased and
fast-forwarded onto `main`, not squashed, so every commit in a PR becomes a changelog line:
before merging, fold review fixups into the commit they fix (`git commit --fixup` and
`git rebase -i --autosquash origin/main`) and word each `feat`/`fix` summary for users.

## Cutting a release

1. Make sure everything meant for the release is merged and CI on `main` is green.

2. Ask git-cliff for the next version. It reads the commits since the last tag: a `feat` (or
   anything breaking) bumps the minor version, otherwise the patch.

   ```bash
   git switch main && git pull
   git cliff --bumped-version          # e.g. v0.3.0
   ```

3. On a branch, set that version in `Cargo.toml` and refresh `Cargo.lock`, then regenerate the
   changelog:

   ```bash
   git switch -c release-0.3.0
   sed -i '0,/^version = /s/^version = .*/version = "0.3.0"/' Cargo.toml
   cargo metadata --format-version 1 >/dev/null   # updates kute's entry in Cargo.lock
   git cliff --tag v0.3.0 -o CHANGELOG.md
   ```

   Read the new section. Each run rewrites the whole file from git, so don't fix a badly worded
   entry in `CHANGELOG.md` by hand: add a rewrite for it to `cliff.toml`'s
   `commit_preprocessors`, and word the next commit better.

4. Commit as `chore(release): v0.3.0`, open a pull request, and rebase and fast-forward it onto
   `main` like any other. The release commit itself is left out of the changelog. If `main`
   moves before it merges, rebase the branch and run step 3's `git cliff` again.

5. Tag the release commit, which is now the tip of `main`, and push the tag:

   ```bash
   git switch main && git pull
   git tag -a v0.3.0 -m "kute 0.3.0"
   git push origin v0.3.0
   ```

   The tag starts `.github/workflows/release.yml`, which checks that the tag matches
   `Cargo.toml` and that `CHANGELOG.md` has the section, builds the portable Ubuntu 24.04
   tarball, and publishes a GitHub release with that section as its notes.

6. Publish the crate to crates.io from the tagged commit, with a token from
   <https://crates.io/settings/tokens> (`cargo login` stores it once):

   ```bash
   cargo publish --dry-run     # packs the crate and builds it from the tarball
   cargo publish
   ```

   A published version can't be replaced or deleted, only yanked, so dry-run first. The
   `include` list in `Cargo.toml` decides what goes into the crate: the code, the icons and the
   bench sample it embeds, the tests and their fixtures, the README, changelog and licenses.
   `cargo package --list` shows the files. The dry run needs the Vulkan headers and `glslc`
   like any default build; `--no-default-features` checks the CPU build without them.

## The prebuilt tarball

`scripts/package-tarball.sh [distro]` builds a release binary with `GGML_NATIVE=OFF` (so it
runs on any x86-64 CPU with AVX2, not only the build machine's) and packs it, with the README,
changelog, licenses and a `SHA256SUMS`, into

```text
dist/kute-<version>-x86_64-<distro>.tar.gz       on a commit tagged v<version>
dist/kute-<version>-<commit>-x86_64-<distro>.tar.gz  anywhere else, or with local changes
```

`distro` defaults to `ubuntu24.04`, which also covers systems based on it, such as Linux
Mint 22. Build on the oldest system the tarball should run on: the binary needs that
system's glibc or newer.

## Fixing a release

Don't move or reuse a tag that's been pushed. Fix on `main` and release a patch version.

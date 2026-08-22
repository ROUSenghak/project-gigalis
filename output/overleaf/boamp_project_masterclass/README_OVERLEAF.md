# BOAMP Project Masterclass - Overleaf package

1. Upload every file in this folder to a new Overleaf project.
2. Set the compiler to **LuaLaTeX**: Menu -> Settings -> Compiler.
3. Set `main.tex` as the main document and click Recompile.

The document uses TeX Gyre fonts distributed with TeX Live and has no network
dependencies. `content.tex` is generated from the canonical teaching guide.

To rebuild the package locally from the repository root:

```bash
python3 scripts/build_project_masterclass_artifact.py
pandoc reports/boamp_project_masterclass.md --from=gfm --to=latex \
  --syntax-highlighting=none --lua-filter=scripts/pandoc_table_widths.lua \
  --output=output/overleaf/boamp_project_masterclass/content.tex
latexmk -cd -lualatex output/overleaf/boamp_project_masterclass/main.tex
```

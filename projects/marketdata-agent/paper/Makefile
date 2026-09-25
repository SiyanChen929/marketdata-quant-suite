# Build the paper from the committed results.
#
#   make          check that the generated tables match results/benchmark/, then build build/main.pdf
#   make tables   regenerate tables/*.tex from results/benchmark/ (never edit them by hand)
#   make check    fail if tables/*.tex are stale
#   make clean    remove LaTeX build products
#
# The PDF needs latexmk and a TeX distribution (pdflatex, bibtex, and the
# booktabs, natbib, tikz, xcolor, geometry, lmodern and hyperref packages).

PYTHON  ?= python3
LATEXMK ?= latexmk
BUILD   := build
RENDER  := ../scripts/render_paper_tables.py

.PHONY: all pdf tables check clean

all: pdf

tables:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) $(RENDER)

check:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) $(RENDER) --check

pdf: check
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -outdir=$(BUILD) main.tex

clean:
	-$(LATEXMK) -C -outdir=$(BUILD) main.tex
	rm -rf $(BUILD)

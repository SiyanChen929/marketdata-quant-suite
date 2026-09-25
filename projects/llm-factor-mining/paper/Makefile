# Build the manuscript with latexmk (pdflatex + bibtex).
#   make          build main.pdf
#   make tables   regenerate tables/*.tex from ../results/synthetic_benchmark/summary.json
#   make check    fail if the committed tables are stale
#   make clean    remove LaTeX intermediates (keeps main.pdf)
#   make distclean  remove intermediates and main.pdf

LATEXMK ?= latexmk
PYTHON ?= python3
MAIN := main

.PHONY: all pdf tables check clean distclean

all: pdf

pdf:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

tables:
	$(PYTHON) ../scripts/render_paper_tables.py

check:
	$(PYTHON) ../scripts/render_paper_tables.py --check

clean:
	$(LATEXMK) -c $(MAIN).tex

distclean:
	$(LATEXMK) -C $(MAIN).tex

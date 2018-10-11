COMPILER = latexmk
OPTIONS = -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make

TEX=$(wildcard *.tex)
PROGS = $(patsubst %.tex,%,$(TEX))
TARGETS = $(patsubst %, %.pdf, $(PROGS))

all: $(TARGETS)

%.pdf: %.tex
	$(COMPILER) $(OPTIONS) $<

clean:
	$(COMPILER) -CA

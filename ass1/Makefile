COMPILER = latexmk
OPTIONS = -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make

TEX=$(wildcard *.tex)
PROGS = $(patsubst %.tex,%,$(TEX))
TARGETS = $(patsubst %, %.pdf, $(PROGS))

all: $(CLEAN) $(TARGETS)

%.pdf: %.tex
	$(COMPILER) $(OPTIONS) $<

clean:
	$(COMPILER) -CA

octave:
	@cd code/ && ./main.m

zip:
	rm *.zip
	zip -r 20316273_octave.zip code/ result_images/ a1.pdf

.PHONY: $(PROGS)


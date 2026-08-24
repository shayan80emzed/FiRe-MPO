MAIN = main

.PHONY: all clean distclean watch wordcount

all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

# Continuous rebuild while writing
watch:
	latexmk -pdf -pvc -interaction=nonstopmode $(MAIN).tex

# Rough body word count (excludes LaTeX markup)
wordcount:
	@for f in chapters/*.tex; do \
		printf "%-28s %s\n" "$$f" "$$(detex $$f 2>/dev/null | wc -w || cat $$f | wc -w)"; \
	done
	@echo "----"
	@printf "%-28s %s\n" "TOTAL" "$$(cat chapters/*.tex | detex 2>/dev/null | wc -w || cat chapters/*.tex | wc -w)"

clean:
	latexmk -c
	rm -f chapters/*.aux

distclean:
	latexmk -C
	rm -f chapters/*.aux

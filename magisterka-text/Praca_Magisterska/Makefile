# -------------------------------------
# WUT-Thesis Makefile build targets
# -------------------------------------

BUILD_DIR = build
PDFS_DIR  = pdfs
MAIN_TEX  = main.tex

MKFLAGS   = -bibtex -pdf -f
TEXFLAGS  = -synctex=1 --interaction=nonstopmode

.PHONY: all base clean watch watch-pdf watch-lua watch-xetex check-latexmk check-pdftex check-luatex check-xetex doctor

# Generate all three PDFs
all: 	pdf lua xetex

# Remove LaTeX auxiliary files
clean:
		rm -f *.app *.aux *.bbl *.bcf *.blg
		rm -f *.fdb_latexmk *.fls *.gz
		rm -f *.lof *.log *.lot *.out *.status
		rm -f *.toc *.xml *-blx.bib *.pdf

		rm -rf $(BUILD_DIR)
		mkdir -p $(BUILD_DIR)
		mkdir -p $(BUILD_DIR)/$(PDFS_DIR)

# Generate LuaLateX PDF
lua:	check-latexmk check-luatex *.tex *.bib ./tex/*
		$(MAKE) base CTEX=lualatex OUT=$(BUILD_DIR)/lualatex

# Generate pdfLaTeX PDF
pdf:	check-latexmk check-pdftex *.tex *.bib ./tex/*
		$(MAKE) base CTEX=pdflatex OUT=$(BUILD_DIR)/pdflatex

# Generate XeLaTeX PDF
xetex:	check-latexmk check-xetex *.tex *.bib ./tex/*
		$(MAKE) base CTEX=xelatex OUT=$(BUILD_DIR)/xelatex

# Make release tarball
release:
		./src/scripts/make-release.sh $(version)

# General building command
base:
		latexmk $(MKFLAGS) \
			-output-directory=$(OUT) \
			-pdflatex="$(CTEX) $(TEXFLAGS)" $(MAIN_TEX)
		mkdir -p $(BUILD_DIR)/$(PDFS_DIR)
		cp -v ./$(OUT)/*.pdf ./$(BUILD_DIR)/$(PDFS_DIR)/main-$(CTEX).pdf

# Continuous build with pdfLaTeX
watch: watch-pdf

# Continuously rebuild PDF on file changes
watch-pdf: check-latexmk check-pdftex
		mkdir -p $(BUILD_DIR)/pdflatex
		latexmk $(MKFLAGS) -pvc -view=none \
			-output-directory=$(BUILD_DIR)/pdflatex \
			-pdflatex="pdflatex $(TEXFLAGS)" $(MAIN_TEX)

# Continuously rebuild PDF on file changes with LuaLaTeX
watch-lua: check-latexmk check-luatex
		mkdir -p $(BUILD_DIR)/lualatex
		latexmk -bibtex -lualatex -f -pvc -view=none \
			-output-directory=$(BUILD_DIR)/lualatex \
			-pdflatex="lualatex $(TEXFLAGS)" $(MAIN_TEX)

# Continuously rebuild PDF on file changes with XeLaTeX
watch-xetex: check-latexmk check-xetex
		mkdir -p $(BUILD_DIR)/xelatex
		latexmk -bibtex -xelatex -f -pvc -view=none \
			-output-directory=$(BUILD_DIR)/xelatex \
			-pdflatex="xelatex $(TEXFLAGS)" $(MAIN_TEX)

check-latexmk:
		@command -v latexmk >/dev/null 2>&1 || { \
			echo "Brak 'latexmk'. Zainstaluj MacTeX albo TeX Live."; \
			exit 127; \
		}

check-pdftex:
		@command -v pdflatex >/dev/null 2>&1 || { \
			echo "Brak 'pdflatex'. Zainstaluj MacTeX albo TeX Live."; \
			exit 127; \
		}

check-luatex:
		@command -v lualatex >/dev/null 2>&1 || { \
			echo "Brak 'lualatex'. Zainstaluj MacTeX albo TeX Live."; \
			exit 127; \
		}

check-xetex:
		@command -v xelatex >/dev/null 2>&1 || { \
			echo "Brak 'xelatex'. Zainstaluj MacTeX albo TeX Live."; \
			exit 127; \
		}

doctor:
		@echo "latexmk: $$(command -v latexmk || echo missing)"
		@echo "pdflatex: $$(command -v pdflatex || echo missing)"
		@echo "lualatex: $$(command -v lualatex || echo missing)"
		@echo "xelatex: $$(command -v xelatex || echo missing)"
